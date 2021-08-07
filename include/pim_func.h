#ifndef PIMTORCH_FUNC_H
#define PIMTORCH_FUNC_H
#pragma once

#include <mutex>
#include <iostream>
#include <cmath>
#include <vector>

using std::vector;
using std::mutex;
using std::lock_guard;
using std::log;
using std::sinh;
using std::fabs;
using std::sqrt;

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

struct operation_count
{
    std::mutex mu;
    void count_add(int64_t x)
    {
        std::lock_guard<std::mutex> lk(mu);
        count+=x;
    }
    operation_count(int x=0): count(x) {}
    int64_t count;
};

//nonlinear unit remains future work
// now the code below is not used
struct nonLinearState
{
    double vBase;
    double G;  // conductance at vBase voltage
    double G_2; // conductance at vBase/2 voltage

    //I = k*sinh(a*v)
    double k, a;

    bool isLinear;

    void set_ak(double a, double k)
    {
        this->a = a;
        this->k = k;
        isLinear = false;
    }

    void set_G2(double v, double g, double g2)
    {
        vBase = v;
        G = g;
        G_2 = g2;
        if (fabs(g-g2)<1e-7)
            isLinear = true;
        else
        {
            isLinear = false;
            set_ak_from_G2();
        }
    }

    double getConductance(double v)
    {
        return isLinear? G : getCurrent(v)/v;
    }

    double getR(double v)
    {
        return isLinear? 1/G : v/getCurrent(v);
    }

    double getI(double v)
    {
        return isLinear? v*G : getCurrent(v);
    }
private:

    double getCurrent(double v)
    {
        return k*sinh(a*v);
    }

    void set_ak_from_G2()
    {
        double tmp, x;
        tmp = 2*G/G_2;
        x = (tmp+sqrt(tmp*tmp-4))/2;
        a = 2*log(x)/vBase;
        k = 2*vBase*G/(exp(a*vBase)-exp(-a*vBase));
    }
};

struct nonLinearUnit
{
    int numStates;  
    vector<nonLinearState> a;
    nonLinearUnit(int s): numStates(s), a(s) {} 
};
#endif