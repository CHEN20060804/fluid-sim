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

       //»æÍ¼Âß¼­
        boundary.drawBoundary(consoleBuffer);
        consoleBuffer.Flush();  // Ë«»º³åäÖÈ¾
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
