#pragma once
#include <vector>
#include <iostream>

namespace lora_kernel {

class WorkStealingScheduler {
private:
    std::vector<std::vector<int>> task_queues;
    int num_threads;
public:
    WorkStealingScheduler(int threads) : num_threads(threads) {
        task_queues.resize(threads);
        std::cout << "[SCHED] Work-stealing scheduler initialized with " 
                  << threads << " threads\n";
    }
    
    void enqueue(int thread_id, int task) {
        task_queues[thread_id].push_back(task);
    }
    
    int steal(int thief_id) {
        for (int i = 0; i < num_threads; ++i) {
            if (i == thief_id) continue;
            if (!task_queues[i].empty()) {
                int task = task_queues[i].back();
                task_queues[i].pop_back();
                return task;
            }
        }
        return -1;
    }
};

} // namespace lora_kernel
