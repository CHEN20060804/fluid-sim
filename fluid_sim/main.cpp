#include <chrono>
#include <thread>
#include "ConsoleBuffer.h"
#include "Boundary.h"
#include <locale>
#include <iostream>
const int gravity = 10;

int main() {
    std::wcout.imbue(std::locale(""));
    ConsoleBuffer consoleBuffer;
    Boundary boundary;
    while (true) {
        consoleBuffer.Clear();
        boundary.drawBoundary(consoleBuffer);
        consoleBuffer.Flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cin.get();
}