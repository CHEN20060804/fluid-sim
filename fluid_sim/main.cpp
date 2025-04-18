#include <chrono>
#include <thread>
#include "ConsoleBuffer.h"
#include <locale>
#include <iostream>

int main() {
    std::wcout.imbue(std::locale(""));
    ConsoleBuffer consoleBuffer;

    while (true) {
        consoleBuffer.Clear();

       //»æÍ¼Âß¼­

        consoleBuffer.Flush();  // Ë«»º³åäÖÈ¾
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
