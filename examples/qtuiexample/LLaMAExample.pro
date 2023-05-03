#-------------------------------------------------
#
# Project created by QtCreator 2023-05-01T11:56:16
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = LLaMAExample
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

INCLUDEPATH += /models/llama.cpp
INCLUDEPATH += /models/llama.cpp/examples
OBJECTS += /models/llama.cpp/common.o /models/llama.cpp/llama.o /models/llama.cpp/ggml.o
