#pragma once
#include <windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <locale>
class ConsoleBuffer {
private:
    int width, height;
    HANDLE hBuffer;
    using Buffer = std::vector<std::wstring>;
    Buffer bufferA;
    Buffer bufferB;

    Buffer* frontBuffer; // 当前显示缓冲区
    Buffer* backBuffer;  // 当前绘图缓冲区

    void HideCursor() {
        CONSOLE_CURSOR_INFO cursorInfo;
        GetConsoleCursorInfo(hBuffer, &cursorInfo);
        cursorInfo.bVisible = FALSE;
        SetConsoleCursorInfo(hBuffer, &cursorInfo);
    }

    void SetConsoleFont() {
        CONSOLE_FONT_INFOEX cfi = { sizeof(cfi) };
        cfi.dwFontSize.X = 4;
        cfi.dwFontSize.Y = 4;
        SetCurrentConsoleFontEx(hBuffer, FALSE, &cfi);
    }

    void SetWindowAndBufferSize(int columns, int rows) {
        COORD bufferSize = { (SHORT)columns, (SHORT)rows };
        SetConsoleScreenBufferSize(hBuffer, bufferSize);
        SMALL_RECT windowSize = { 0, 0, columns - 1, rows - 1 };
        SetConsoleWindowInfo(hBuffer, TRUE, &windowSize);
    }

    void MaximizeConsoleWindow() {
        HWND hwnd = GetConsoleWindow();
        ShowWindow(hwnd, SW_MAXIMIZE);
    }

public:
    ConsoleBuffer(int columns = 400, int rows = 300)
        : width(columns), height(rows),
        bufferA(rows, std::wstring(columns, L' ')),
        bufferB(rows, std::wstring(columns, L' ')) {

        frontBuffer = &bufferA;
        backBuffer = &bufferB;

        hBuffer = CreateConsoleScreenBuffer(
            GENERIC_READ | GENERIC_WRITE,
            0, NULL, CONSOLE_TEXTMODE_BUFFER, NULL);

        SetConsoleFont();
        SetWindowAndBufferSize(columns, rows);
        SetConsoleActiveScreenBuffer(hBuffer);

        HideCursor();
        MaximizeConsoleWindow();
        SetStdHandle(STD_OUTPUT_HANDLE, hBuffer);
    }

    void Clear()
    {
        std::fill(backBuffer->begin(), backBuffer->end(), std::wstring(width, L' '));
    }


    void DrawAt(int x, int y, wchar_t c) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            (*backBuffer)[y][x] = c;
        }
    }

    void SwapBuffers() {
        std::swap(frontBuffer, backBuffer);
    }

    void Render() {
        std::vector<CHAR_INFO> charBuffer(width * height);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                wchar_t ch = (*frontBuffer)[y][x];
                CHAR_INFO& ci = charBuffer[y * width + x];
                ci.Char.UnicodeChar = ch;
                ci.Attributes = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE; // 白色
            }
        }

        COORD bufferSize = { (SHORT)width, (SHORT)height };
        COORD bufferCoord = { 0, 0 };
        SMALL_RECT writeRegion = { 0, 0, (SHORT)(width - 1), (SHORT)(height - 1) };

        WriteConsoleOutputW(
            hBuffer,
            charBuffer.data(),
            bufferSize,
            bufferCoord,
            &writeRegion
        );
    }

    void Flush() {
        SwapBuffers();  // 交换前后缓冲区
        Render();  // 渲染缓冲区内容
    }
    int GetWidth() const { return width; }
    int GetHeight() const { return height; }
};
