// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "/models/llama.cpp/examples/common.h"
#include "/models/llama.cpp/llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QTimer>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

static console_state con_st;
static llama_context ** g_ctx;

static bool is_interacting = false;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    initializeLLaMA();
}

MainWindow::~MainWindow()
{
    delete ui;
}


#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    //set_console_color(con_st, CONSOLE_COLOR_DEFAULT);
   printf("\n"); // this also force flush stdout.
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting=true;
        } else {
            llama_print_timings(*g_ctx);
            _exit(130);
        }
    }
}
#endif

void MainWindow::initializeLLaMA() {

    params.model = "/models/llama.cpp/models/13B/ggml-model-q4_0.bin";


    params.n_ctx = 2048;


    params.seed = time(NULL);
    params.n_predict = 80;
    params.prompt = "This is a test prompt.";
    ui->initialPromptLineEdit->setText(QString::fromStdString(params.prompt));
    ui->nTokensGenLineEdit->setText(QString("%1").arg((int)params.n_predict));
    ui->temperatureLineEdit->setText(QString("%1").arg(params.temp));
//    ui->plainTextEdit->appendPlainText(QString( "%s: seed = %d\n", __func__, params.seed);
    ui->plainTextEdit->appendPlainText(QString("%1: seed = %2\n").arg(__func__).arg(params.seed));

    std::mt19937 rng(params.seed);

//    llama_context * ctx;
    g_ctx = &ctx;

    // load the model
    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx      = params.n_ctx;
        lparams.n_parts    = params.n_parts;
        lparams.seed       = params.seed;
        lparams.f16_kv     = params.memory_f16;
        lparams.use_mmap   = params.use_mmap;
        lparams.use_mlock  = params.use_mlock;

        ctx = llama_init_from_file(params.model.c_str(), lparams);

        if (ctx == NULL) {
//            ui->plainTextEdit->appendPlainText(QString( "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            ui->plainTextEdit->appendPlainText(QString("%1: error: failed to load model '%2'\n").arg(__func__).arg(params.model.c_str()));
            return ;
        }
    }


    // print system information
    {
        ui->plainTextEdit->appendPlainText(QString("\n"));

        ui->plainTextEdit->appendPlainText(QString("system_info: n_threads = %1 / %2 | %3\n").arg(
                params.n_threads).arg( std::thread::hardware_concurrency()).arg( llama_print_system_info()));
    }

    // determine the maximum memory usage needed to do inference for the given n_batch and n_predict parameters
    // uncomment the "used_mem" line in llama.cpp to see the results
    if (params.mem_test) {
        {
            const std::vector<llama_token> tmp(params.n_batch, 0);
            llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
        }

        {
            const std::vector<llama_token> tmp = { 0, };
            llama_eval(ctx, tmp.data(), tmp.size(), params.n_predict - 1, params.n_threads);
        }

        llama_print_timings(ctx);
        llama_free(ctx);

        return ;
    }

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    path_session = params.path_session;

    if (!path_session.empty()) {
        ui->plainTextEdit->appendPlainText(QString("%1: attempting to load saved session from '%2'\n").arg( __func__).arg( path_session.c_str()));

        // fopen to check for existing session
        FILE * fp = std::fopen(path_session.c_str(), "rb");
        if (fp != NULL) {
            std::fclose(fp);

            session_tokens.resize(params.n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_load_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                ui->plainTextEdit->appendPlainText(QString("%1: error: failed to load session file '%2'\n").arg( __func__).arg(path_session.c_str()));
                return ;
            }
            session_tokens.resize(n_token_count_out);

            ui->plainTextEdit->appendPlainText(QString("%1: loaded a session with prompt size of %2 tokens\n").arg( __func__).arg( (int) session_tokens.size()));
        } else {
            ui->plainTextEdit->appendPlainText(QString("%1: session file does not exist, will create\n").arg( __func__));
        }
    }

    // tokenize the prompt
    embd_inp = ::llama_tokenize(ctx, params.prompt, true);

    n_ctx = llama_n_ctx(ctx);

    if ((int) embd_inp.size() > n_ctx - 4) {
        ui->plainTextEdit->appendPlainText(QString("%1: error: prompt is too long (%2 tokens, max %3)\n").arg( __func__).arg( (int) embd_inp.size()).arg( n_ctx - 4));
        return ;
    }
    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (session_tokens.size()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (n_matching_session_tokens >= embd_inp.size()) {
            ui->plainTextEdit->appendPlainText(QString( "%1: session file has exact match for prompt!\n").arg( __func__));
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            ui->plainTextEdit->appendPlainText(QString( "%1: warning: session file has low similarity to prompt (%2 / %3 tokens); will mostly be reevaluated\n").arg(
                __func__).arg( n_matching_session_tokens).arg( embd_inp.size()));
        } else {
            ui->plainTextEdit->appendPlainText(QString( "%1: session file matches %2 / %3 tokens of prompt\n")
                .arg(__func__).arg(n_matching_session_tokens).arg( embd_inp.size()));
        }
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct) {
        params.n_keep = (int)embd_inp.size();
    }

    // prefix & suffix for instruct mode
    inp_pfx = ::llama_tokenize(ctx, "\n\n### Instruction:\n\n", true);
    inp_sfx = ::llama_tokenize(ctx, "\n\n### Response:\n\n", false);

    // in instruct mode, we inject a prefix and a suffix to each input by the user
    if (params.instruct) {
        params.interactive_first = true;
        params.antiprompt.push_back("### Instruction:\n\n");
    }

    // enable interactive mode if reverse prompt or interactive start is specified
    if (params.antiprompt.size() != 0 || params.interactive_first) {
        params.interactive = true;
    }

    // determine newline token
    llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    if (params.verbose_prompt) {
        ui->plainTextEdit->appendPlainText(QString( "\n"));
        ui->plainTextEdit->appendPlainText(QString( "%1: prompt: '%2'\n").arg( __func__).arg( params.prompt.c_str()));
        ui->plainTextEdit->appendPlainText(QString( "%1: number of tokens in prompt = %2\n").arg( __func__).arg( embd_inp.size()));
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            ui->plainTextEdit->appendPlainText(QString( "%1 -> '%2'\n").arg( embd_inp[i]).arg( llama_token_to_str(ctx, embd_inp[i])));
        }
        if (params.n_keep > 0) {
        ui->plainTextEdit->appendPlainText(QString( "%1: static prompt based on n_keep: '").arg( __func__));
            for (int i = 0; i < params.n_keep; i++) {
                ui->plainTextEdit->appendPlainText(QString( "%1").arg(llama_token_to_str(ctx, embd_inp[i])));
            }
            ui->plainTextEdit->appendPlainText(QString( "'\n"));
        }
        ui->plainTextEdit->appendPlainText(QString( "\n"));
    }

    if (params.interactive) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        signal(SIGINT, sigint_handler);
#endif

        ui->plainTextEdit->appendPlainText(QString( "%1: interactive mode on.\n").arg( __func__));

        if (params.antiprompt.size()) {
            for (auto antiprompt : params.antiprompt) {
                ui->plainTextEdit->appendPlainText(QString( "Reverse prompt: '%1'\n").arg( antiprompt.c_str()));
            }
        }

        if (!params.input_prefix.empty()) {
            ui->plainTextEdit->appendPlainText(QString( "Input prefix: '%1'\n").arg(params.input_prefix.c_str()));
        }
    }
    ui->plainTextEdit->appendPlainText(QString( "sampling: repeat_last_n = %1, repeat_penalty = %2, presence_penalty = %3, frequency_penalty = %4, top_k = %5, tfs_z = %6, top_p = %7, typical_p = %8, temp = %9, mirostat = %10, mirostat_lr = %11, mirostat_ent = %12\n").arg(
            params.repeat_last_n).arg( params.repeat_penalty).arg( params.presence_penalty).arg( params.frequency_penalty).arg( params.top_k).arg( params.tfs_z).arg( params.top_p).arg( params.typical_p).arg( params.temp).arg( params.mirostat).arg( params.mirostat_eta).arg( params.mirostat_tau));
    ui->plainTextEdit->appendPlainText(QString( "generate: n_ctx = %1, n_batch = %2, n_predict = %3, n_keep = %4\n").arg( n_ctx).arg( params.n_batch).arg( params.n_predict).arg( params.n_keep));
    ui->plainTextEdit->appendPlainText(QString( "\n\n"));

    // TODO: replace with ring-buffer
    last_n_tokens.reserve(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    if (params.interactive) {
        ui->plainTextEdit->appendPlainText(QString( "== Running in interactive mode. ==\n"
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
               " - Press Ctrl+C to interject at any time.\n"
#endif
               " - Press Return to return control to LLaMa.\n"
               " - If you want to submit another line, end your input in '\\'.\n\n"));
        is_interacting = params.interactive_first;
    }

    is_antiprompt = false;
    input_noecho  = false;

    // HACK - because session saving incurs a non-negligible delay, for now skip re-saving session
    // if we loaded a session with at least 75% similarity. It's currently just used to speed up the
    // initial prompt so it doesn't need to be an exact match.
    need_to_save_session = !path_session.empty() && n_matching_session_tokens < (embd_inp.size() * 3 / 4);


    n_past     = 0;
    n_remain   = params.n_predict;
    n_consumed = 0;
    n_session_consumed = 0;

    // the first thing we will do is to output the prompt, so set color accordingly
    set_console_color(con_st, CONSOLE_COLOR_PROMPT);

//    std::vector<llama_token> embd;
   //  QTimer::singleShot(200, this, &MainWindow::on_pushButton_clicked );
}


int MainWindow::interactivePrompt(){
    ui->pushButton->setDisabled(true);
    while (n_remain != 0 || params.interactive) {
        printf("%d",n_remain);
        // predict
        if (embd.size() > 0) {
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int) embd.size() > n_ctx) {
                const int n_left = n_past - params.n_keep;

                n_past = params.n_keep;

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());

                // stop saving session if we run out of context
                path_session = "";

            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            // REVIEW
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }
                if (llama_eval(ctx, &embd[i], n_eval, n_past, params.n_threads)) {
                    ui->plainTextEdit_2->appendPlainText(QString( "%s : failed to eval\n").arg( __func__));
                    return 1;
                }
                n_past += n_eval;
            }

            if (embd.size() > 0 && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // out of user input, sample next token
            const float   temp            = params.temp;
            const int32_t top_k           = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
            const float   top_p           = params.top_p;
            const float   tfs_z           = params.tfs_z;
            const float   typical_p       = params.typical_p;
            const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
            const float   repeat_penalty  = params.repeat_penalty;
            const float   alpha_presence  = params.presence_penalty;
            const float   alpha_frequency = params.frequency_penalty;
            const int     mirostat        = params.mirostat;
            const float   mirostat_tau    = params.mirostat_tau;
            const float   mirostat_eta    = params.mirostat_eta;
            const bool    penalize_nl     = params.penalize_nl;

            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session) {
                need_to_save_session = false;
                llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
            }

            llama_token id = 0;

            {
                auto logits = llama_get_logits(ctx);
                auto n_vocab = llama_n_vocab(ctx);

                // Apply params.logit_bias map
                for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                    logits[it->first] += it->second;
                }

                std::vector<llama_token_data> candidates;
                candidates.reserve(n_vocab);
                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                }

                llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                // Apply penalties
                float nl_logit = logits[llama_token_nl()];
                auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
                llama_sample_repetition_penalty(ctx, &candidates_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, repeat_penalty);
                llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, alpha_frequency, alpha_presence);
                if (!penalize_nl) {
                    logits[llama_token_nl()] = nl_logit;
                }

                if (temp <= 0) {
                    // Greedy sampling
                    id = llama_sample_token_greedy(ctx, &candidates_p);
                } else {
                    if (mirostat == 1) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        const int mirostat_m = 100;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                    } else if (mirostat == 2) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                    } else {
                        // Temperature sampling
                        llama_sample_top_k(ctx, &candidates_p, top_k);
                        llama_sample_tail_free(ctx, &candidates_p, tfs_z);
                        llama_sample_typical(ctx, &candidates_p, typical_p);
                        llama_sample_top_p(ctx, &candidates_p, top_p);
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token(ctx, &candidates_p);
                    }
                }
                // printf("`%d`", candidates_p.size);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // replace end of text token with newline token when in interactive mode
            if (id == llama_token_eos() && params.interactive && !params.instruct) {
                id = llama_token_newline.front();
                if (params.antiprompt.size() != 0) {
                    // tokenize and inject first reverse prompt
                    const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false);
                    embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                }
            }

            // add it to the context
            embd.push_back(id);

            // echo this to console
            input_noecho = false;

            // decrement remaining sampling budget
            --n_remain;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (!input_noecho) {
            for (auto id : embd) {
                ui->plainTextEdit_2->insertPlainText(QString("%1").arg( llama_token_to_str(ctx, id)));
                printf("%s",llama_token_to_str(ctx,id));
                QCoreApplication::processEvents();
            }
            fflush(stdout);
        }
        // reset color to default if we there is no pending user input
        if (!input_noecho && (int)embd_inp.size() == n_consumed) {
            set_console_color(con_st, CONSOLE_COLOR_DEFAULT);
        }

        // in interactive mode, and not currently processing queued inputs;
        // check if we should prompt the user for more
        if (params.interactive && (int) embd_inp.size() <= n_consumed) {

            // check for reverse prompt
            if (params.antiprompt.size()) {
                std::string last_output;
                for (auto id : last_n_tokens) {
                    last_output += llama_token_to_str(ctx, id);
                }

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                for (std::string & antiprompt : params.antiprompt) {
                    if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
                        is_interacting = true;
                        is_antiprompt = true;
                        set_console_color(con_st, CONSOLE_COLOR_USER_INPUT);
                        fflush(stdout);
                        break;
                    }
                }
            }

            if (n_past > 0 && is_interacting) {
                // potentially set color to indicate we are taking user input
                set_console_color(con_st, CONSOLE_COLOR_USER_INPUT);

#if defined (_WIN32)
                // Windows: must reactivate sigint handler after each signal
                signal(SIGINT, sigint_handler);
#endif

                if (params.instruct) {
                    ui->plainTextEdit_2->appendPlainText(QString("\n> "));
                }

                std::string buffer;
                if (!params.input_prefix.empty()) {
                    buffer += params.input_prefix;
                    ui->plainTextEdit_2->appendPlainText(QString("%1").arg( buffer.c_str()));
                }

                std::string line;
                bool another_line = true;
                do {
#if defined(_WIN32)
                    std::wstring wline;
                    if (!std::getline(std::wcin, wline)) {
                        // input stream is bad or EOF received
                        return 0;
                    }
                    win32_utf8_encode(wline, line);
#else
                    if (!std::getline(std::cin, line)) {
                        // input stream is bad or EOF received
                        return 0;
                    }
#endif
                    if (line.empty() || line.back() != '\\') {
                        another_line = false;
                    } else {
                        line.pop_back(); // Remove the continue character
                    }
                    buffer += line + '\n'; // Append the line to the result
                } while (another_line);

                // done taking input, reset color
                set_console_color(con_st, CONSOLE_COLOR_DEFAULT);

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {

                    // instruct mode: insert instruction prefix
                    if (params.instruct && !is_antiprompt) {
                        n_consumed = embd_inp.size();
                        embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
                    }

                    auto line_inp = ::llama_tokenize(ctx, buffer, false);
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

                    // instruct mode: insert response suffix
                    if (params.instruct) {
                        embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                    }

                    n_remain -= line_inp.size();
                }

                input_noecho = true; // do not echo this again
            }

            if (n_past > 0) {
                is_interacting = false;
            }
        }

        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos()) {
            if (params.instruct) {
                is_interacting = true;
            } else {
                ui->plainTextEdit_2->appendPlainText_2(QString( " [end of text]\n"));
                ui->pushButton->setEnabled(true);
                return 0;
            }
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        if (params.interactive && n_remain <= 0 && params.n_predict != -1) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }
    ui->pushButton->setEnabled(true);
    return 0;
}


void MainWindow::on_pushButton_clicked()
{
    processInput();
}

void MainWindow::processInput()
{
    n_remain   = params.n_predict;
    std::string buffer = ui->lineEdit->text().toStdString();
    if (buffer.length() > 1) {

        // instruct mode: insert instruction prefix
        if (params.instruct && !is_antiprompt) {
            n_consumed = embd_inp.size();
            embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
        }

        auto line_inp = ::llama_tokenize(ctx, buffer, false);
        embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

        // instruct mode: insert response suffix
        if (params.instruct) {
            embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
        }

        n_remain -= line_inp.size();
    }
    interactivePrompt();
}

void MainWindow::on_lineEdit_returnPressed()
{
    processInput();
}

void MainWindow::on_nTokensGenLineEdit_editingFinished()
{
    params.n_predict = ui->nTokensGenLineEdit->text().toInt();
}

void MainWindow::on_temperatureLineEdit_editingFinished()
{
    params.temp = ui->temperatureLineEdit->text().toFloat();
}

void MainWindow::on_initialPromptLineEdit_editingFinished()
{
    params.prompt = ui->initialPromptLineEdit->text().toStdString();

    embd_inp = ::llama_tokenize(ctx, params.prompt, true);

    n_ctx = llama_n_ctx(ctx);

    if ((int) embd_inp.size() > n_ctx - 4) {
        ui->plainTextEdit->appendPlainText(QString("%1: error: prompt is too long (%2 tokens, max %3)\n").arg( __func__).arg( (int) embd_inp.size()).arg( n_ctx - 4));
        return ;
    }
}
