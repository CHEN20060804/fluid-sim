#include <chrono>
#include <thread>
#include "ConsoleBuffer.h"
#include "utility.h"
#include "Particle.h"
#include "Integrator.h"
#define FPS 60
 const int gravity = 10;
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
    DrawRectangleFrame(consoleBuffer, frameLeft, frameTop, frameRight, frameBottom);
	Particle particle0;
    particle0.position = Vec2((frameLeft + frameRight) / 2, (frameTop + frameBottom) / 2);
    particle0.velocity = Vec2(0, 0);
	particle0.force = Vec2(0, 10);
	std::vector<Particle> particles;
	particles.push_back(particle0);
	
    
    auto lastDrawTime = std::chrono::steady_clock::now();

    while (true) {
        auto currentTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = currentTime - lastDrawTime;
        if (elapsed.count() >= 1.0/FPS) {

            consoleBuffer.Clear();
            DrawRectangleFrame(consoleBuffer, frameLeft, frameTop, frameRight, frameBottom);
            Integrator::step(particles, 0.016f, 1.0f);
			DrawAt(consoleBuffer, particles[0].position.getX(), particles[0].position.getY(), L'●');
            
            consoleBuffer.Render();

            lastDrawTime = currentTime;
        }
        else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    return 0;
}