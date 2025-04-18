#include <chrono>
#include <thread>
#include "ConsoleBuffer.h"
#include "Boundary.h"
#include <locale>
#include <iostream>

int main() {
    std::wcout.imbue(std::locale(""));
    ConsoleBuffer consoleBuffer;
    Boundary boundary;
    while (true) {
        consoleBuffer.Clear();

       //��ͼ�߼�
        boundary.drawBoundary(consoleBuffer);
        consoleBuffer.Flush();  // ˫������Ⱦ
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
