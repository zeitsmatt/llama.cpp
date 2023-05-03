#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "../common.h"
#include "../../llama-util.h"
namespace Ui {
class MainWindow;
}
class MainWindow : public QMainWindow
{
    Q_OBJECT
    gpt_params params;
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void initializeLLaMA();
    void processInput();
    int interactivePrompt();

private slots:
    void on_pushButton_clicked();

    void on_lineEdit_returnPressed();

    void on_nTokensGenLineEdit_editingFinished();

    void on_temperatureLineEdit_editingFinished();

    void on_initialPromptLineEdit_editingFinished();

private:
    Ui::MainWindow *ui;
    std::vector<llama_token> embd;
    std::vector<int32_t> embd_inp;
    int n_ctx;
    int n_past;
    int n_remain;
    int n_consumed;
    int n_session_consumed;
    bool is_antiprompt;
    llama_context * ctx;
    bool input_noecho;
    bool need_to_save_session;
    std::vector<llama_token> last_n_tokens;
    std::vector<llama_token>  llama_token_newline;
    std::vector<llama_token>  inp_pfx;
    std::vector<llama_token>  inp_sfx;
    std::string path_session = params.path_session;
    std::vector<llama_token> session_tokens;

};
/*class PromptWorker : public QObject {
    Q_OBJECT
public:
    PromptWorker();
    ~PromptWorker();
public slots:
    void process();
signals:
    void finished();
    void error(QString err);
private:
};*/

#endif // MAINWINDOW_H
