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
    const float maxWindForce = 5.0f;  // �߽�����������
    const float windRadius = 200.0f;   // ���Ӱ��뾶
    const float upwardForce = 3.0f;  // �ո�������
    const float leftBoundary = 1.0f;   // ��߽�λ��
    const float rightBoundary = 920.0f; // �ұ߽�λ��

    auto getWindForce = [=](float distanceFromBoundary) -> float {
        if (distanceFromBoundary < windRadius) {
            return maxWindForce * (1.0f - (distanceFromBoundary / windRadius));
        }
        return 0.0f;
        };

    // ������
    std::for_each(std::execution::par, particles.begin(), particles.end(), [&](Particle& p) {
        const float m = p.mass;
        if (spacePressed) {
            if (300 < p.position.getX() < 600)
            {
                float distanceToCenter = std::abs(p.position.getX() - (rightBoundary + leftBoundary) * 0.5f);
                float maxDistance = (rightBoundary - leftBoundary) * 0.5f;
                float reductionFactor = 1.0f - (distanceToCenter / maxDistance); // �м���1����Ե��0

                reductionFactor = std::max(0.0f, reductionFactor); // ���ף���ֹ����

                p.velocity -= 0.5 * Vec2(0, upwardForce * reductionFactor) / m;
            }
        }

        if (leftAltPressed) {
            if (p.position.getX() < windRadius) {
                float distanceFromLeftBoundary = p.position.getX() - leftBoundary;
                float appliedWindForce = getWindForce(distanceFromLeftBoundary);
                p.velocity = p.velocity + Vec2(0, -appliedWindForce) / m;  // ��������
            }
        }

        if (rightAltPressed) {
            if (p.position.getX() > (rightBoundary - windRadius)) {
                float distanceFromRightBoundary = -p.position.getX() + rightBoundary;
                float appliedWindForce = getWindForce(distanceFromRightBoundary);
                p.velocity = p.velocity + Vec2(0, -appliedWindForce) / m; // ��������
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
                    particle->velocity *= 0.95; // Ӧ������
                }

            }
            lock.lock();
        }
    }
}
void Manager::solveOverlop(float minDist)
{
    float maxSeparation = 0.3f;   // �������������
    float restitution = 1.01f;     // ���Իָ�ϵ��

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

                // λ��������ÿ��������һ��
                pi.position += dir * separation;
                pj.position -= dir * separation;

                // �򵥵����ٶ�����
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
    std::cout << "��ѡ��ģʽ������0ΪSPHģʽ��1ΪPBFģʽ������󰴻س���" << std::endl;
    std::cin >> model;
    if (model == SPH)
        std::cout << "��ѡ��SPHģʽ���������ʼ��������[���飨1000<=nums<=10000)����󰴻س�]" << std::endl;
    else if (model == PBF)
        std::cout << "��ѡ��PBFģʽ���������ʼ��������[���飨8000<=nums<=13000)����󰴻س�]" << std::endl;
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
    std::cout << "��ӭ����  [����ī��]\n\n";
    std::cout << "����һ������ SPH��Smoothed Particle Hydrodynamics���� PBF��Position-Based Fluids�������Ķ�ά����ģ������\n";
    std::cout << "ϵͳͨ��������ɢ����ʽģ���������ʵĶ���ѧ��Ϊ��\n\n\n";
    std::cout << "˵����\n--------------------------------------------------------------\n";
    std::cout << "����ģ��ǰ��\n  Ϊ��������Ⱦ���ӣ���ʼ��Ϸǰ�����������Ϸ� �� ��ť����� [����] ��\n  �ҵ� [Ĭ���ն�Ӧ�ó���] ѡ���������Ϊ [Windows����̨����]��\n  ��� [����] ���˳������´򿪳���";
    std::cout << "\n--------------------------------------------------------------\n";
    std::cout << "����ģ���\n";
    std::cout << "1  ��ס [��Alt��] �������½Ǵ����\n";
    std::cout << "2  ��ס [�ո�] �����в������\n";
    std::cout << "3  ��ס [��Alt��] �������½Ǵ����\n";
    std::cout << "4  ��ס [������] ��������\n";
    std::cout << "5  ��ס [����Ҽ�] �ų�����\n";
    std::cout << "\n--------------------------------------------------------------\n";
    std::cout << "���س�����..." << std::endl;
    std::cin.get();
    std::cout << std::endl;
}
