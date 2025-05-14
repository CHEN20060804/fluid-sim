#pragma once
#include <windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <locale>
#include "Particle.h"
class ConsoleBuffer {
private:
    int width, height;
    HANDLE hBuffer;
    using Buffer = std::vector<std::wstring>;
    Buffer bufferA;
    Buffer bufferB;

    std::vector<std::vector<WORD>> colorBuffer;

    Buffer* frontBuffer; // 当前显示缓冲区
    Buffer* backBuffer;  // 当前绘图缓冲区

    void HideCursor();

    void SetConsoleFont();

    void SetWindowAndBufferSize(int columns, int rows);

    void MaximizeConsoleWindow();

public:

    ConsoleBuffer(int columns = 1500, int rows = 1500);

    void Clear();

    void DrawAt(int x, int y, wchar_t c);

    void DrawAt(int x, int y, const wchar_t& c, WORD color);


    void drawParticles(std::vector<Particle>& particles);

    void SwapBuffers();

    void Render();

    void Flush();

    int GetWidth() const;

    int GetHeight() const;
};
