#include <chrono>
#include <thread>
#include "ConsoleBuffer.h"
#include <locale>


int main() {
    std::wcout.imbue(std::locale(""));
    // ����˫�������
    ConsoleBuffer consoleBuffer;
    // ���ع��
    HideCursor();
    if (!IsRunningAsAdmin()) {
        wchar_t exePath[MAX_PATH];
        GetModuleFileNameW(NULL, exePath, MAX_PATH);

        SHELLEXECUTEINFOW sei = { sizeof(sei) };
        sei.lpVerb = L"runas";
        sei.lpFile = exePath;
        sei.nShow = SW_SHOWNORMAL;

        if (!ShellExecuteExW(&sei)) {
            std::cerr << "��Ȩʧ�ܣ������룺" << GetLastError() << std::endl;
            return 1;
        }
        return 0;
    }
    int frameLeft, frameTop, frameRight, frameBottom;

    while (true) {
        consoleBuffer.Clear();
        DrawRectangleFrame(consoleBuffer, frameLeft, frameTop, frameRight, frameBottom);
        consoleBuffer.Render();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}