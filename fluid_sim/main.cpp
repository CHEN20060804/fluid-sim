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

       //��ͼ�߼�

        consoleBuffer.Flush();  // ˫������Ⱦ
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
