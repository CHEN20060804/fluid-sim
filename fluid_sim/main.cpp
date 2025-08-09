#include <chrono>
#include <thread>
#include "ConsoleBuffer.h"
#include "Boundary.h"
#include <locale>
#include <iostream>
#include "Particle.h"
#include "SPH.h"
#include <fstream>
#include "PBF.h"
#include <random>
#include <conio.h>
#include <functional>
#include "Manager.h"

int main() {
    int nums = 0;
    int model = -1;
    std::vector<Particle> particles;
    Manager manager(particles);

  
    manager.showInstructions();
    while (true)
    {
        std::thread producter(&Manager::listenMouseInput, &manager);
        std::thread consumer(&Manager::handleMouseEvents, &manager);
        std::thread listenKeyboard(&Manager::listenKeyboardEvents, &manager);
        std::thread handleKeyboard(&Manager::handleKeyboardEvents, &manager, std::ref(particles));
        manager.readyForGame(nums, model);
        if (!manager.enterGame(particles, nums, model))
        {
            std::cout << "无效的模式选择！重新初始化。" << std::endl;
            continue;
        }
        ParticleGrid::getInstance().init(particles);

        ConsoleBuffer consoleBuffer;
        Boundary boundary;

        while (true)
        {
            consoleBuffer.Clear();
            manager.mainSolve();
            consoleBuffer.drawParticles(particles);
            boundary.drawBoundary(consoleBuffer);
            consoleBuffer.Flush();
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }
        manager.over();
        producter.join();
        consumer.join();
        listenKeyboard.join();
        handleKeyboard.join();
    }
   

    std::cin.get();
}



