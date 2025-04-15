#include <chrono>
#include <thread>
#include "ConsoleBuffer.h"
#include "utility.h"
#define FPS 60

int main() {
    std::wcout.imbue(std::locale(""));
    // 创建双缓冲对象
    ConsoleBuffer consoleBuffer;
    // 隐藏光标
    HideCursor();
    if (!IsRunningAsAdmin()) {
        wchar_t exePath[MAX_PATH];
        GetModuleFileNameW(NULL, exePath, MAX_PATH);

        SHELLEXECUTEINFOW sei = { sizeof(sei) };
        sei.lpVerb = L"runas";
        sei.lpFile = exePath;
        sei.nShow = SW_SHOWNORMAL;

        if (!ShellExecuteExW(&sei)) {
            std::cerr << "提权失败，错误码：" << GetLastError() << std::endl;
            return 1;
        }
        return 0;
    }
    int frameLeft=0, frameTop=0, frameRight=0, frameBottom=0;
    
    auto lastDrawTime = std::chrono::steady_clock::now();

    Vec2 position((frameLeft + frameRight) / 2, (frameTop + frameBottom) / 2);
    Vec2 velocity(0, 0);

    while (true) {
        auto currentTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = currentTime - lastDrawTime;
        if (elapsed.count() >= 1.0/FPS) {

            consoleBuffer.Clear();
            velocity += Vec2(0, 1.0f) * 5 * 1 / FPS;
			position += velocity * 1 / FPS;
			if (position.getY() > consoleBuffer.GetHeight() - 1) {
				position.setY(consoleBuffer.GetHeight() - 1);
				velocity.setY(-velocity.getY());
			}
			DrawAt(consoleBuffer, position.getX(), position.getY(), L'●');
            DrawRectangleFrame(consoleBuffer, frameLeft, frameTop, frameRight, frameBottom);
            consoleBuffer.Render();

            lastDrawTime = currentTime;
        }
        else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    return 0;
}