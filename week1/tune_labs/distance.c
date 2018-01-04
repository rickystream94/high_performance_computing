#include "distance.h"
#include <unistd.h>
#include <math.h>

#ifdef ALL_IN_ONE

double
distance(particle_t *p, int n)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < n; i++)
    {
        double r = sqrt(exp(p[i].x) + exp(p[i].y) + exp(p[i].z));
        p[i].dist = r;
        sum += r;
    }
    return sum;
}

#else

double
distance(particle_t *p, double *v, int n)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < n; i++)
    {
        double r = sqrt(exp(p[i].x) + exp(p[i].y) + exp(p[i].z));
        sum += r;
        v[i] = r;
    }
    return sum;
}

#endif
