#include "distcheck.h"
#include <unistd.h>

#ifdef ALL_IN_ONE

double
distcheck(particle_t *p, int n)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < n; i++)
    {
        sum += p[i].dist;
    }
    return sum;
}

#else

double
distcheck(double *v, int n)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < n; i++)
    {
        sum += v[i];
    }
    return sum;
}

#endif
