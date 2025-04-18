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
        wcscpy_s(cfi.FaceName, L"Lucida Console");
        cfi.FontFamily = FF_DONTCARE;
        cfi.FontWeight = FW_NORMAL;
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
    ConsoleBuffer(int columns = 120, int rows = 40)
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
        SetConsoleActiveScreenBuffer(hBuffer);//显示自定义的控制台窗口
        HideCursor();
        MaximizeConsoleWindow();
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
        DWORD charsWritten;

        // 创建一个包含所有字符的字符串
        std::wstring fullBuffer;
        for (int y = 0; y < height; ++y) {
            fullBuffer.append((*frontBuffer)[y]);
        }

        // 一次性写入整个缓冲区
        WriteConsoleOutputCharacterW(
            hBuffer,
            fullBuffer.c_str(),
            fullBuffer.length(),
            { 0, 0 },  // 从屏幕左上角开始写入
            &charsWritten
        );
    }

    void Flush() {
        SwapBuffers();
        Render();
    }

    int GetWidth() const { return width; }
    int GetHeight() const { return height; }
};
