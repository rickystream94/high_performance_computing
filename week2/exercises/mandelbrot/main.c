#include <stdio.h>
#include <stdlib.h>
#include "mandel.h"
#include "writepng.h"
#include <omp.h>

int main(int argc, char *argv[])
{

    int width, height;
    int max_iter;
    int *image;

    width = 2601;
    height = 2601;
    max_iter = 400;

    // command line argument sets the dimensions of the image
    if (argc == 2)
        width = height = atoi(argv[1]);

    image = (int *)malloc(width * height * sizeof(int));
    if (image == NULL)
    {
        fprintf(stderr, "memory allocation failed!\n");
        return (1);
    }

    // Count start time
    double ts, te;
    ts = omp_get_wtime();
    mandel(width, height, image, max_iter);
    te = omp_get_wtime() - ts;

    writepng("mandelbrot.png", image, width, height);

    printf("%f\n", te);

    return (0);
}
