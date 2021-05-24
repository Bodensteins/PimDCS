#ifndef PIMTORCH_FUNC_H
#define PIMTORCH_FUNC_H
#pragma once

#include <mutex>
#include <iostream>
#include <cmath>

using std::mutex;
using std::lock_guard;

template <typename T>
inline T trunc_ceil(T x, T mod)
{
    return (x + mod - 1) / mod;
}

/**
 * this struct is used to count latency
 * a thread safe counter
 * */
struct pim_latency
{
    mutex mu;
    double run_latency_us; 
    double run_latency_s;

    pim_latency(double s = 0, double us = 0): run_latency_s(s), run_latency_us(us) {}
    
    void latency_add(double time_ns)
    {
        lock_guard<mutex> lk(mu);
        latency_add_without_lck(time_ns);
    }

    void latency_add(pim_latency &other)
    {
        lock_guard<mutex> lk(mu);
        other.mu.lock();
        run_latency_s += other.run_latency_s;
        run_latency_us += other.run_latency_us;
        other.mu.unlock();
        latency_add_without_lck(0);
    }

    void inline latency_add_without_lck(double time_ns)
    {
        run_latency_us += time_ns / 1000.0;
        double add = floor(run_latency_us /1e6);

        run_latency_s += add;
        run_latency_us -= add*1e6;
    }

    void print_latency(std::ostream &os)
    {
        os << run_latency_s << " (s) " << run_latency_us << " (us)";
    }
};
#endif