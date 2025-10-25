#include <bits/stdc++.h>
#include <chrono>
#include <thread>
using namespace std;

static long long work(int n){
    // simple CPU-bound workload: multiply-accumulate + sqrt
    double acc = 0;
    for(int i=1;i<=n;i++){
        acc += sqrt((double)i) * 1.000001;
    }
    return (long long)acc;
}

int main(int argc, char** argv){
    int total_iterations = 90000000;  // Fixed workload targeting ~30ms single-thread on typical PC
    int threads = 1;
    if(argc >= 2) total_iterations = atoi(argv[1]);
    if(argc >= 3) threads = atoi(argv[2]);

    // Distribute work among threads
    int iterations_per_thread = total_iterations / threads;
    
    vector<thread> ts;
    auto t0 = chrono::high_resolution_clock::now();
    for(int t=0;t<threads;t++){
        ts.emplace_back([&](){ work(iterations_per_thread); });
    }
    for(auto& th: ts) th.join();
    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, std::milli>(t1 - t0).count();
    cout << "wall_ms=" << ms << ",iterations=" << total_iterations
         << ",threads=" << threads << endl;
    return 0;
}
