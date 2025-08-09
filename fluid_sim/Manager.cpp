#define NOMINMAX
#include "Manager.h"
#include <random>
#include "Boundary.h"
#include "pbf.cuh"
Manager::Manager(std::vector<Particle>& particles)
    :sph(particles), running(true)
{

}
void Manager::applyWindEffect(std::vector<Particle>& particles, bool spacePressed, bool leftAltPressed, bool rightAltPressed) {
    const float maxWindForce = 5.0f;  // 边界风机的最大风力
    const float windRadius = 200.0f;   // 风机影响半径
    const float upwardForce = 3.0f;  // 空格向上力
    const float leftBoundary = 1.0f;   // 左边界位置
    const float rightBoundary = 920.0f; // 右边界位置

    auto getWindForce = [=](float distanceFromBoundary) -> float {
        if (distanceFromBoundary < windRadius) {
            return maxWindForce * (1.0f - (distanceFromBoundary / windRadius));
        }
        return 0.0f;
        };

    // 开并行
    std::for_each(std::execution::par, particles.begin(), particles.end(), [&](Particle& p) {
        const float m = p.mass;
        if (spacePressed) {
            if (300 < p.position.getX() < 600)
            {
                float distanceToCenter = std::abs(p.position.getX() - (rightBoundary + leftBoundary) * 0.5f);
                float maxDistance = (rightBoundary - leftBoundary) * 0.5f;
                float reductionFactor = 1.0f - (distanceToCenter / maxDistance); // 中间是1，边缘是0

                reductionFactor = std::max(0.0f, reductionFactor); // 保底，防止负数

                p.velocity -= 0.5 * Vec2(0, upwardForce * reductionFactor) / m;
            }
        }

        if (leftAltPressed) {
            if (p.position.getX() < windRadius) {
                float distanceFromLeftBoundary = p.position.getX() - leftBoundary;
                float appliedWindForce = getWindForce(distanceFromLeftBoundary);
                p.velocity = p.velocity + Vec2(0, -appliedWindForce) / m;  // 向右上推
            }
        }

        if (rightAltPressed) {
            if (p.position.getX() > (rightBoundary - windRadius)) {
                float distanceFromRightBoundary = -p.position.getX() + rightBoundary;
                float appliedWindForce = getWindForce(distanceFromRightBoundary);
                p.velocity = p.velocity + Vec2(0, -appliedWindForce) / m; // 向左上推
            }
        }
        });
}

void Manager::listenKeyboardEvents()
{
    while (running)
    {
        if (GetAsyncKeyState(VK_SPACE) & 0x8000)
        {
            std::lock_guard<std::mutex> lock(KeyboardQueueMutex);
            keyboardEventQueue.push({ VK_SPACE,true });
        }
        else
        {
            std::lock_guard<std::mutex> lock(KeyboardQueueMutex);
            keyboardEventQueue.push({ VK_SPACE,false });
        }
        if (GetAsyncKeyState(VK_LMENU) & 0x8000) {
            std::lock_guard<std::mutex> lock(KeyboardQueueMutex);
            keyboardEventQueue.push({ VK_LMENU, true });
        }
        else {
            std::lock_guard<std::mutex> lock(KeyboardQueueMutex);
            keyboardEventQueue.push({ VK_LMENU, false });
        }
        if (GetAsyncKeyState(VK_RMENU) & 0x8000) {
            std::lock_guard<std::mutex> lock(KeyboardQueueMutex);
            keyboardEventQueue.push({ VK_RMENU, true });
        }
        else {
            std::lock_guard<std::mutex> lock(KeyboardQueueMutex);
            keyboardEventQueue.push({ VK_RMENU, false });
        }
        if (GetAsyncKeyState(VK_TAB) & 0x8000) {
            std::lock_guard<std::mutex> lock(KeyboardQueueMutex);
            keyboardEventQueue.push({ VK_TAB, true });
        }
        else {
            std::lock_guard<std::mutex> lock(KeyboardQueueMutex);
            keyboardEventQueue.push({ VK_TAB, false });
        }
        KeyboardEventCondition.notify_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }
}
void Manager::handleKeyboardEvents(std::vector<Particle>& particles)
{
    while (running)
    {
        std::unique_lock<std::mutex> lock(KeyboardQueueMutex);
        KeyboardEventCondition.wait(lock, [&] { return !keyboardEventQueue.empty() || !running; });

        if (!running) break;

        while (!keyboardEventQueue.empty())
        {
            auto [keyCode, isPressed] = keyboardEventQueue.front();
            keyboardEventQueue.pop();
            lock.unlock();
            if (keyCode == VK_SPACE) {
                if (isPressed) {
                    applyWindEffect(particles, true, false, false);
                }
            }
            else if (keyCode == VK_LMENU) {
                if (isPressed) {
                    applyWindEffect(particles, false, true, false);
                }
            }
            else if (keyCode == VK_RMENU) {
                if (isPressed) {
                    applyWindEffect(particles, false, false, true);
                }
            }
            lock.lock();
        }
    }
}
void Manager::enableMouseInput()
{
    hInput = GetStdHandle(STD_INPUT_HANDLE);
    GetConsoleMode(hInput, &previousMode);

    DWORD newMode = previousMode & ~ENABLE_QUICK_EDIT_MODE;
    newMode |= ENABLE_MOUSE_INPUT | ENABLE_EXTENDED_FLAGS;

    SetConsoleMode(hInput, newMode);
}
void Manager::listenMouseInput()
{
    enableMouseInput();

    INPUT_RECORD record;
    DWORD events;

    while (running) {
        ReadConsoleInput(hInput, &record, 1, &events);
        if (record.EventType == MOUSE_EVENT) {
            COORD pos = record.Event.MouseEvent.dwMousePosition;
            {
                std::lock_guard<std::mutex> lock(MouseQueueMutex);
                if (record.Event.MouseEvent.dwButtonState & FROM_LEFT_1ST_BUTTON_PRESSED) {
                    mouseEventQueue.push({ pos, 1 });
                }
                if (record.Event.MouseEvent.dwButtonState & RIGHTMOST_BUTTON_PRESSED) {
                    mouseEventQueue.push({ pos, 2 });
                }
            }
            MouseEventCondition.notify_one();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }
}
void Manager::handleMouseEvents()
{
    while (running)
    {
        std::unique_lock<std::mutex> lock(MouseQueueMutex);

        MouseEventCondition.wait(lock, [&] { return !mouseEventQueue.empty() || !running; });

        if (!running) break;

        while (!mouseEventQueue.empty())
        {
            std::unique_lock<std::mutex> lock(particlesMutex);

            auto [pos, eventType] = mouseEventQueue.front();
            mouseEventQueue.pop();
            lock.unlock();

            Vec2 clickPos(pos.X, pos.Y);
            auto ans = ParticleGrid::getInstance().ForeachMouseWithinRadius(clickPos, 60.0f);

            std::vector<std::pair<Particle*, Vec2>> toUpdate;

            for (Particle& p : ans)
            {
                if (!p.isSolid)
                {
                    Vec2 Dir = clickPos - p.position;
                    if (eventType == 2)
                        Dir = -Dir;
                    float distance = Dir.length();

                    if (distance > 1e-5)
                    {
                        float dampingFactor = 0.98f;
                        Dir *= (1.0f / distance);
                        toUpdate.push_back({ &p, Dir * 15.0f });
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lock(particlesMutex);
                for (auto& [particle, velocity] : toUpdate) {
                    particle->velocity += velocity;
                    particle->velocity *= 0.95; // 应用阻尼
                }

            }
            lock.lock();
        }
    }
}
void Manager::solveOverlop(float minDist)
{
    float maxSeparation = 0.3f;   // 单次最大分离距离
    float restitution = 1.01f;     // 弹性恢复系数

    size_t count = sph.particles.size();

    for (size_t i = 0; i < count; ++i) {
        Particle& pi = sph.particles[i];

        for (size_t j = i + 1; j < count; ++j) {
            Particle& pj = sph.particles[j];

            Vec2 delta = pi.position - pj.position;
            float dist = delta.length();

            if (dist < minDist && dist > 1e-5f) {
                Vec2 dir = delta / dist; // normalize
                float overlap = minDist - dist;
                float separation = std::min(overlap * 0.5f, maxSeparation);

                // 位置修正：每个粒子移一半
                pi.position += dir * separation;
                pj.position -= dir * separation;

                // 简单弹性速度修正
                Vec2 relativeVelocity = pi.velocity - pj.velocity;
                float vAlongNormal = relativeVelocity * dir;

                if (vAlongNormal < 0.0f) {
                    float impulseMag = -(1 + restitution) * vAlongNormal * 0.5f;
                    Vec2 impulse = dir * impulseMag;
                    pi.velocity += impulse;
                    pj.velocity -= impulse;
                }
            }
        }
    }
}
void Manager::generateRandomParticles_SPH(std::vector<Particle>& particles, int count, float xMin, float xMax, float yMin, float yMax) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distX(xMin, xMax);
    std::uniform_real_distribution<float> distY(yMin, yMax);

    for (int i = 0; i < count; ++i) {
        float x = distX(gen);
        float y = distY(gen);
        particles.emplace_back(Particle(Vec2(x, y)));
    }
}
void Manager::generateRandomParticles_PBF(std::vector<Particle>& particles, int count, float xMin, float xMax, float yMin, float yMax) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distX(xMin, xMax);
    std::uniform_real_distribution<float> distY(yMin, yMax);

    for (int i = 0; i < count; ++i) {
        float x = distX(gen);
        float y = distY(gen);
        particles.emplace_back(Particle(Vec2(x, y)));
        particles[i].index = i;
    }

    for (int i = 0; i < count; ++i)
    {
        SPHSolver sphSolver(particles);
        sphSolver.computeDensityForPBF(particles[i], particles);
    }
}
bool Manager::enterGame(std::vector<Particle>& particles, int& num, int& model)
{
    if (model == SPH)
    {
        generateRandomParticles_SPH(particles, num, 50, WIDTH - 50, 200, 350);
        mainSolve = [&]() {
            sph.simulateStep(0.05f); };
        return true;
    }
    else if (model == PBF)
    {
        generateRandomParticles_PBF(particles, num, 50, WIDTH - 50, 200, 350);
        mainSolve = [&]()
        {solve(particles, 0.05, ParticleGrid::getInstance().spatiacleLookat, ParticleGrid::getInstance().startIndex);
        solveOverlop(8.0f); 
        };
        return true;
    }
    return false;
}
void Manager::readyForGame(int& nums, int& model)
{
    std::cout << "请选择模式（输入0为SPH模式，1为PBF模式，输入后按回车）" << std::endl;
    std::cin >> model;
    if (model == SPH)
        std::cout << "已选择SPH模式，请输入初始粒子数量[建议（1000<=nums<=10000)输入后按回车]" << std::endl;
    else if (model == PBF)
        std::cout << "已选择PBF模式，请输入初始粒子数量[建议（8000<=nums<=13000)输入后按回车]" << std::endl;
    else return;
    std::cin >> nums;
}
void Manager::over()
{
    running = false;
    MouseEventCondition.notify_all();
    KeyboardEventCondition.notify_all();
}
void Manager::showInstructions() {
    std::cout << "欢迎来到  [流体墨客]\n\n";
    std::cout << "这是一个基于 SPH（Smoothed Particle Hydrodynamics）和 PBF（Position-Based Fluids）方法的二维流体模拟器。\n";
    std::cout << "系统通过粒子离散化方式模拟连续介质的动力学行为。\n\n\n";
    std::cout << "说明：\n--------------------------------------------------------------\n";
    std::cout << "进入模拟前：\n  为了正常渲染粒子，开始游戏前请点击窗口最上方 ∨ 按钮，点击 [设置] ，\n  找到 [默认终端应用程序] 选项，将其设置为 [Windows控制台主机]，\n  点击 [保存] 后退出，重新打开程序。";
    std::cout << "\n--------------------------------------------------------------\n";
    std::cout << "进入模拟后：\n";
    std::cout << "1  按住 [左Alt键] 开启左下角吹风机\n";
    std::cout << "2  按住 [空格] 开启中部吹风机\n";
    std::cout << "3  按住 [右Alt键] 开启右下角吹风机\n";
    std::cout << "4  按住 [鼠标左键] 吸引粒子\n";
    std::cout << "5  按住 [鼠标右键] 排斥粒子\n";
    std::cout << "\n--------------------------------------------------------------\n";
    std::cout << "按回车继续..." << std::endl;
    std::cin.get();
    std::cout << std::endl;
}
