#include <Windows.h>
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

    Buffer* frontBuffer; // ��ǰ��ʾ������
    Buffer* backBuffer;  // ��ǰ��ͼ������

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
        SetConsoleActiveScreenBuffer(hBuffer);//��ʾ�Զ���Ŀ���̨����
        HideCursor();
        MaximizeConsoleWindow();
    }

    void Clear() {
        for (auto& line : *backBuffer) {
            line.assign(width, L' ');
        }
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
        for (int y = 0; y < height; ++y) {
            WriteConsoleOutputCharacterW(
                hBuffer,
                (*frontBuffer)[y].c_str(),
                (*frontBuffer)[y].length(),
                { 0, (SHORT)y },
                &charsWritten
            );
        }
    }

    void Flush() {
        SwapBuffers();
        Render();
    }

    int GetWidth() const { return width; }
    int GetHeight() const { return height; }
};
