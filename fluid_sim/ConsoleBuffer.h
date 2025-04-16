#pragma once
#include <iostream>
#include <vector>
#include <Windows.h>
#include <locale>

class ConsoleBuffer {
private:
    int width, height;
    std::vector<std::wstring> buffer;
    HANDLE hConsole;

public:
    ConsoleBuffer() {
        hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

        HWND hwnd = GetConsoleWindow();
        ShowWindow(hwnd, SW_MAXIMIZE);

        SetConsoleFont();

        CONSOLE_SCREEN_BUFFER_INFO csbi;
        GetConsoleScreenBufferInfo(hConsole, &csbi);

        COORD largestSize = GetLargestConsoleWindowSize(hConsole);

        // 设置窗口缓冲区为最大可见尺寸
        SetConsoleScreenBufferSize(hConsole, largestSize);

        // 设置窗口矩形为满缓冲区大小
        SMALL_RECT windowSize = { 0, 0, largestSize.X - 1, largestSize.Y - 1 };
        SetConsoleWindowInfo(hConsole, TRUE, &windowSize);

        // 更新本地记录的宽高
        width = largestSize.X;
        height = largestSize.Y;

        // 初始化缓冲区
        buffer = std::vector<std::wstring>(height, std::wstring(width, L' '));
    }

    void UpdateWindowSize() {
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        GetConsoleScreenBufferInfo(hConsole, &csbi);

        width = csbi.srWindow.Right - csbi.srWindow.Left + 1;
        height = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
    }

    void Clear() {
        for (auto& line : buffer) {
            line.assign(width, L' ');
        }
    }

    void DrawAt(int x, int y, wchar_t c) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            buffer[y][x] = c;
        }
    }

    void Render() {
        DWORD charsWritten;
        for (int y = 0; y < height; ++y) {
            WriteConsoleOutputCharacterW(
                hConsole,
                buffer[y].c_str(),
                buffer[y].length(),
                { 0, (SHORT)y },
                &charsWritten
            );
        }
    }
    int GetWidth() const {
        return width;
    }
    int GetHeight() const {
        return height;
    }
    void SetConsoleFont() {
        CONSOLE_FONT_INFOEX cfi = { sizeof(cfi) };
        cfi.dwFontSize.X = 8;
        cfi.dwFontSize.Y = 16;
        wcscpy_s(cfi.FaceName, L"Lucida Console");
        cfi.FontFamily = FF_DONTCARE;
        cfi.FontWeight = FW_NORMAL;
        SetCurrentConsoleFontEx(hConsole, FALSE, &cfi);
    }
    void SetPixel(int x, int y, wchar_t sign) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            buffer[y][x] = sign; 
        }
    }

};

void DrawRectangleFrame(ConsoleBuffer& consoleBuffer, int& frameLeft, int& frameTop, int& frameRight, int& frameBottom) {
    int width = consoleBuffer.GetWidth();
    int height = consoleBuffer.GetHeight();

    int boxWidth = width * 0.8;
    int boxHeight = height * 0.8;

    int startX = (width - boxWidth) / 2;
    int startY = (height - boxHeight) / 2;

    frameLeft = startX;
    frameTop = startY;
    frameRight = startX + boxWidth - 1;
    frameBottom = startY + boxHeight - 1;

    for (int y = 0; y < boxHeight; ++y) {
        for (int x = 0; x < boxWidth; ++x) {
            if (y == 0 && x == 0) consoleBuffer.DrawAt(startX + x, startY + y, L'┌');
            else if (y == 0 && x == boxWidth - 1) consoleBuffer.DrawAt(startX + x, startY + y, L'┐');
            else if (y == boxHeight - 1 && x == 0) consoleBuffer.DrawAt(startX + x, startY + y, L'└');
            else if (y == boxHeight - 1 && x == boxWidth - 1) consoleBuffer.DrawAt(startX + x, startY + y, L'┘');
            else if (y == 0 || y == boxHeight - 1) consoleBuffer.DrawAt(startX + x, startY + y, L'─');
            else if (x == 0 || x == boxWidth - 1) consoleBuffer.DrawAt(startX + x, startY + y, L'│');
        }
    }
}
void MoveCursorTo(int x, int y) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD pos = { (SHORT)x, (SHORT)y };
    SetConsoleCursorPosition(hConsole, pos);
}
void HideCursor() {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_CURSOR_INFO cursorInfo;
    GetConsoleCursorInfo(hConsole, &cursorInfo);
    cursorInfo.bVisible = FALSE;
    SetConsoleCursorInfo(hConsole, &cursorInfo);
}
bool IsRunningAsAdmin() {
    BOOL isAdmin = FALSE;
    SID_IDENTIFIER_AUTHORITY NtAuthority = SECURITY_NT_AUTHORITY;
    PSID adminGroup;
    if (AllocateAndInitializeSid(&NtAuthority, 2,
        SECURITY_BUILTIN_DOMAIN_RID, DOMAIN_ALIAS_RID_ADMINS,
        0, 0, 0, 0, 0, 0, &adminGroup)) {
        CheckTokenMembership(NULL, adminGroup, &isAdmin);
        FreeSid(adminGroup);
    }
    return isAdmin;
}

void DrawAt(ConsoleBuffer& consoleBuffer, int x, int y, wchar_t sign) {
    consoleBuffer.SetPixel(x, y, sign);  // 将符号绘制到缓冲区中的指定位置
}