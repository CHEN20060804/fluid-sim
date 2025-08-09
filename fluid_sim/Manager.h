#pragma once
#include <Windows.h>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <thread>
#include <chrono>
#include "ParticleGrid.h"
#include "utility.h"
#include "SPH.h"
//#include "pbf.cuh"
#include <execution>
#include <functional>
enum Model
{
    SPH = 0,
    PBF = 1
};
class Manager {
private:
    SPHSolver sph;
    double lastAddTime = 0.0;
    DWORD previousMode;
    HANDLE hInput;
    std::mutex particlesMutex;
    std::mutex MouseQueueMutex;
    std::mutex KeyboardQueueMutex;
    std::condition_variable MouseEventCondition;
    std::condition_variable KeyboardEventCondition;
    std::queue<std::pair<COORD, int>> mouseEventQueue;
    struct KeyboardState {
        int keyCode;
        bool isPressed;
    };

    KeyboardState keyboardState;
    std::queue<KeyboardState> keyboardEventQueue;
    bool running;

public:
    std::function<void()> mainSolve;

    Manager(std::vector<Particle>& particles);

    bool enterGame(std::vector<Particle>& particles, int& num, int& model);

    void readyForGame(int& nums, int& model);

    void applyWindEffect(std::vector<Particle>& particles, bool spacePressed, bool leftAltPressed, bool rightAltPressed);

    void listenKeyboardEvents();

    void handleKeyboardEvents(std::vector<Particle>& particles);

    void enableMouseInput();

    void listenMouseInput();

    void handleMouseEvents();

	void solveOverlop(float minDist);

    void generateRandomParticles_SPH(std::vector<Particle>& particles, int count, float xMin, float xMax, float yMin, float yMax);

    void generateRandomParticles_PBF(std::vector<Particle>& particles, int count, float xMin, float xMax, float yMin, float yMax);

    void over();

    void showInstructions();
};