#include "ConsoleBuffer.h"
void ConsoleBuffer::HideCursor() {
    CONSOLE_CURSOR_INFO cursorInfo;
    GetConsoleCursorInfo(hBuffer, &cursorInfo);
    cursorInfo.bVisible = FALSE;
    SetConsoleCursorInfo(hBuffer, &cursorInfo);
}

void ConsoleBuffer::SetConsoleFont() {
    CONSOLE_FONT_INFOEX cfi = { sizeof(cfi) };
    cfi.dwFontSize.X = 2;
    cfi.dwFontSize.Y = 2;
    SetCurrentConsoleFontEx(hBuffer, FALSE, &cfi);
}

void ConsoleBuffer::SetWindowAndBufferSize(int columns, int rows) {
    COORD bufferSize = { (SHORT)columns, (SHORT)rows };
    SetConsoleScreenBufferSize(hBuffer, bufferSize);
    SMALL_RECT windowSize = { 0, 0, columns - 1, rows - 1 };
    SetConsoleWindowInfo(hBuffer, TRUE, &windowSize);
}

void ConsoleBuffer::MaximizeConsoleWindow() {
    HWND hwnd = GetConsoleWindow();
    ShowWindow(hwnd, SW_MAXIMIZE);
}

ConsoleBuffer::ConsoleBuffer(int columns, int rows)
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

    colorBuffer = std::vector(height, std::vector<WORD>(width, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE));
}

void ConsoleBuffer::Clear()
{
    std::fill(backBuffer->begin(), backBuffer->end(), std::wstring(width, L' '));
}


void ConsoleBuffer::DrawAt(int x, int y, wchar_t c) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        (*backBuffer)[y][x] = c;
    }
}

void ConsoleBuffer::DrawAt(int x, int y, const wchar_t& c, WORD color) {
    if (y >= 0 && y < height) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            (*backBuffer)[y][x] = c;
            colorBuffer[y][x] = color;
        }
    }
}

void ConsoleBuffer::drawParticles(std::vector<Particle>& particles)
{
    for (int i = 0; i < particles.size(); i++) {
        int x = static_cast<int>(std::round(particles[i].position.getX()));
        int y = static_cast<int>(std::round(particles[i].position.getY()));
        
            DrawAt(x, y, L'●', FOREGROUND_BLUE);
        
    }//●
}
void ConsoleBuffer::SwapBuffers() {
    std::swap(frontBuffer, backBuffer);
}

void ConsoleBuffer::Render() {
    std::vector<CHAR_INFO> charBuffer(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            wchar_t ch = (*frontBuffer)[y][x];
            CHAR_INFO& ci = charBuffer[y * width + x];
            ci.Char.UnicodeChar = ch;
            ci.Attributes = colorBuffer[y][x]; // 白色
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

void ConsoleBuffer::Flush() {
    SwapBuffers();  // 交换前后缓冲区
    Render();  // 渲染缓冲区内容
}

int ConsoleBuffer::GetWidth() const { return width; }
int ConsoleBuffer::GetHeight() const { return height; }