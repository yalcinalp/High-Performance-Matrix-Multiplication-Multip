/********************************************************
 * Kernels to be optimized for the CS:APP Performance Lab
 ********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
/*
 * team_t struct
 */
team_t team = {
        "venom",    /* Team Name */

        "e2522126",    /* First student ID */
        "Alp Eren Yalcin",    /* First student name */
        
        "e2521268",        /* Second student ID */
        "Akın Aydemir"         /* Second student name */

};

/************************
 * POINT REFLECTION KERNEL
 ************************/

/*********************************************************
 * Different versions of the point reflection go here
 *********************************************************/

/*
 * naive_reflect - The naive baseline version of point reflection
 */
char naive_reflect_descr[] = "Naive Point Reflection: Naive baseline implementation";
void naive_reflect(int dim, int *src, int *dst) {
    int i, j;

    for (i = 0; i < dim; i++) {
	for (j = 0; j < dim; j++) {
	    dst[RIDX(dim-1-i, dim-1-j, dim)] = src[RIDX(i, j, dim)];
	}
    }
}


char fast_reflect_descr[] = "fast Point Reflection: fast baseline implementation";
void fast_reflect(int dim, int *src, int *dst) {
    int i;
    int n = dim*dim;
    for (i = 0; i < n; i+= 8)
    {
            dst[n - i - 1] = src[i];
            dst[n - i - 2] = src[i + 1];
            dst[n - i - 3] = src[i + 2];
            dst[n - i - 4] = src[i + 3];
            dst[n - i - 5] = src[i + 4];
            dst[n - i - 6] = src[i + 5];
            dst[n - i - 7] = src[i + 6];
            dst[n - i - 8] = src[i + 7];
    }
}

/*
 * reflect - Your current working version of reflect
 */
char reflect_descr[] = "Point Reflection: Current working version";
void reflect(int dim, int *src, int *dst)
{
    fast_reflect(dim,src,dst);
}

void register_reflect_functions() {
    add_reflect_function(&naive_reflect, naive_reflect_descr);
    add_reflect_function(&reflect, reflect_descr);
    add_reflect_function(&fast_reflect, fast_reflect_descr);
    /* ... Register additional test functions here */
}

/*******************************************************
 * BATCHED MATRIX MULTIPLICATION \W SUM REDUCTION KERNEL
 *******************************************************/

/*********************************************************************************
 * Different versions of the batched matrix multiplication functions go here
 *********************************************************************************/

/*
 * naive_batched_mm - The naive baseline version of batched matrix multiplication
 */
char naive_batched_mm_descr[] = "naive_batched_mm: Naive baseline implementation";
void naive_batched_mm(int dim, int *b_mat, int *mat, int *dst) {
    int i,j,k,l;
    
    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            for (k = 0; k < dim; k++) {
            	if (i == 0){
            	    dst[RIDX(j, k, dim)] = 0;
            	}
            	for (l = 0; l < dim; l++){
                    dst[RIDX(j, k, dim)] += b_mat[RIDX(i*dim+j, l, dim)] * mat[RIDX(l, k, dim)];
                }
            }
        }
    }
}


char fast_batched_mm_descr[] = "fast_batched_mm: fast batch";
void fast_batched_mm(int dim, int *b_mat, int *mat, int *dst) {
    int i,j,k,l;
    int dimSquare = dim * dim;
    int dimCube = dimSquare * dim;

    int *temp = (int *) malloc(dimSquare * sizeof(int));

    // first clear dst matrix
    for (i = 0; i < dimSquare; i+=32) {
        dst[i] = 0;
    }

    // sum all matrices on temp
    for (i = 0; i < dim; i++)
    {
        for (j = 0; j < dim; j++)
        {
            for (k = 0; k < dim; k++)
            {
                temp[i * dim + j] += b_mat[k * dimSquare + i * dim + j];
            }
        }
    }

    for (int i = 0; i < dim; i++) {         // ith row, jth column 
        for (int j = 0; j < dim; j++) {      
            for (int k = 0; k < dim; k++) { 
                dst[i * dim + j] += temp[i * dim + k] * mat[k * dim + j];
            }
        }
    }
}



char faster_batched_mm_descr[] = "fasterr_batched_mm: fasterr batch";
void faster_batched_mm(int dim, int *b_mat, int *mat, int *dst) {
    int i,j,k,l, s;
    int dimSquare = dim * dim;
    int dimCube = dimSquare * dim;

    int *temp = (int *) calloc(dimSquare, sizeof(int));

    // first clear dst matrix
    for (i = 0; i < dimSquare; i += 32) {
        dst[i] = 0;
        dst[i + 1] = 0;
        dst[i + 2] = 0;
        dst[i + 3] = 0;
        dst[i + 4] = 0;
        dst[i + 5] = 0;
        dst[i + 6] = 0;
        dst[i + 7] = 0;
        dst[i + 8] = 0;
        dst[i + 9] = 0;
        dst[i + 10] = 0;
        dst[i + 11] = 0;
        dst[i + 12] = 0;
        dst[i + 13] = 0;
        dst[i + 14] = 0;
        dst[i + 15] = 0;
        dst[i + 16] = 0;
        dst[i + 17] = 0;
        dst[i + 18] = 0;
        dst[i + 19] = 0;
        dst[i + 20] = 0;
        dst[i + 21] = 0;
        dst[i + 22] = 0;
        dst[i + 23] = 0;
        dst[i + 24] = 0;
        dst[i + 25] = 0;
        dst[i + 26] = 0;
        dst[i + 27] = 0;
        dst[i + 28] = 0;
        dst[i + 29] = 0;
        dst[i + 30] = 0;
        dst[i + 31] = 0;
    }

    // sum all matrices on temp
    for (s = 0; s < dim; s++)
    {
        k = s * dimSquare;
        for (i = 0; i < dimSquare; i+=32)
        {
            temp[i] += b_mat[k];
            temp[i + 1] += b_mat[k + 1];
            temp[i + 2] += b_mat[k + 2];
            temp[i + 3] += b_mat[k + 3];
            temp[i + 4] += b_mat[k + 4];
            temp[i + 5] += b_mat[k + 5];
            temp[i + 6] += b_mat[k + 6];
            temp[i + 7] += b_mat[k + 7];
            temp[i + 8] += b_mat[k + 8];
            temp[i + 9] += b_mat[k + 9];
            temp[i + 10] += b_mat[k + 10];
            temp[i + 11] += b_mat[k + 11];
            temp[i + 12] += b_mat[k + 12];
            temp[i + 13] += b_mat[k + 13];
            temp[i + 14] += b_mat[k + 14];
            temp[i + 15] += b_mat[k + 15];
            temp[i + 16] += b_mat[k + 16];
            temp[i + 17] += b_mat[k + 17];
            temp[i + 18] += b_mat[k + 18];
            temp[i + 19] += b_mat[k + 19];
            temp[i + 20] += b_mat[k + 20];
            temp[i + 21] += b_mat[k + 21]; 
            temp[i + 22] += b_mat[k + 22];
            temp[i + 23] += b_mat[k + 23];
            temp[i + 24] += b_mat[k + 24];
            temp[i + 25] += b_mat[k + 25];
            temp[i + 26] += b_mat[k + 26];
            temp[i + 27] += b_mat[k + 27];
            temp[i + 28] += b_mat[k + 28];
            temp[i + 29] += b_mat[k + 29];
            temp[i + 30] += b_mat[k + 30];
            temp[i + 31] += b_mat[k + 31];
            k += 32;
        }
    }

    for (int i = 0; i < dim; i++) {         // ith row, jth column 
        for (int j = 0; j < dim; j++) {      
            for (int k = 0; k < dim; k++) { 
                dst[i * dim + j] += temp[i * dim + k] * mat[k * dim + j];
            }
        }
    }
}


char way_faster_batched_mm_descr[] = "way_fasterr_batched_mm: way fasterr venom batch";
void way_faster_batched_mm(int dim, int *b_mat, int *mat, int *dst) {
    int i,j,k,l,s,r, p, sj, pj;
    int dimSquare = dim * dim;
    int dimCube = dimSquare * dim;

    int *temp = (int *) calloc(dimSquare, sizeof(int));

    // first clear dst matrix
    for (i = 0; i < dimSquare; i += 32) {
        dst[i] = 0;
        dst[i + 1] = 0;
        dst[i + 2] = 0;
        dst[i + 3] = 0;
        dst[i + 4] = 0;
        dst[i + 5] = 0;
        dst[i + 6] = 0;
        dst[i + 7] = 0;
        dst[i + 8] = 0;
        dst[i + 9] = 0;
        dst[i + 10] = 0;
        dst[i + 11] = 0;
        dst[i + 12] = 0;
        dst[i + 13] = 0;
        dst[i + 14] = 0;
        dst[i + 15] = 0;
        dst[i + 16] = 0;
        dst[i + 17] = 0;
        dst[i + 18] = 0;
        dst[i + 19] = 0;
        dst[i + 20] = 0;
        dst[i + 21] = 0;
        dst[i + 22] = 0;
        dst[i + 23] = 0;
        dst[i + 24] = 0;
        dst[i + 25] = 0;
        dst[i + 26] = 0;
        dst[i + 27] = 0;
        dst[i + 28] = 0;
        dst[i + 29] = 0;
        dst[i + 30] = 0;
        dst[i + 31] = 0;
    }    
    // sum all matrices on temp
    for (s = 0; s < dim; s++)
    {
        k = s * dimSquare;
        for (i = 0; i < dimSquare; i+=32)
        {
            temp[i] += b_mat[k];
            temp[i + 1] += b_mat[k + 1];
            temp[i + 2] += b_mat[k + 2];
            temp[i + 3] += b_mat[k + 3];
            temp[i + 4] += b_mat[k + 4];
            temp[i + 5] += b_mat[k + 5];
            temp[i + 6] += b_mat[k + 6];
            temp[i + 7] += b_mat[k + 7];
            temp[i + 8] += b_mat[k + 8];
            temp[i + 9] += b_mat[k + 9];
            temp[i + 10] += b_mat[k + 10];
            temp[i + 11] += b_mat[k + 11];
            temp[i + 12] += b_mat[k + 12];
            temp[i + 13] += b_mat[k + 13];
            temp[i + 14] += b_mat[k + 14];
            temp[i + 15] += b_mat[k + 15];
            temp[i + 16] += b_mat[k + 16];
            temp[i + 17] += b_mat[k + 17];
            temp[i + 18] += b_mat[k + 18];
            temp[i + 19] += b_mat[k + 19];
            temp[i + 20] += b_mat[k + 20];
            temp[i + 21] += b_mat[k + 21]; 
            temp[i + 22] += b_mat[k + 22];
            temp[i + 23] += b_mat[k + 23];
            temp[i + 24] += b_mat[k + 24];
            temp[i + 25] += b_mat[k + 25];
            temp[i + 26] += b_mat[k + 26];
            temp[i + 27] += b_mat[k + 27];
            temp[i + 28] += b_mat[k + 28];
            temp[i + 29] += b_mat[k + 29];
            temp[i + 30] += b_mat[k + 30];
            temp[i + 31] += b_mat[k + 31];
            k += 32;
        }
    }
             
    // mult - w loop unrolling
    p = -dim;
    for (k = 0; k < dim; k++)
    {
	p += dim;
	s = -dim;
        for (i = 0; i < dim; i++)
        {
	    s += dim;
            r = temp[s + k];
            for(j = 0; j < dim; j += 8)
            {
                sj = s + j;
                pj = p + j;
                dst[sj]     += r * mat[pj];
                dst[sj + 1] += r * mat[pj + 1];
                dst[sj + 2] += r * mat[pj + 2];
                dst[sj + 3] += r * mat[pj + 3];
                dst[sj + 4] += r * mat[pj + 4];
                dst[sj + 5] += r * mat[pj + 5];
                dst[sj + 6] += r * mat[pj + 6];
                dst[sj + 7] += r * mat[pj + 7];
            }
        }
    }      
     
}


/*
 * batched_mm - Current working version of batched matrix multiplication
 */
char batched_mm_descr[] = "Batched MM with sum reduction: Current working version";
void batched_mm(int dim, int *b_mat, int *mat, int *dst)
{

    way_faster_batched_mm(dim,b_mat,mat,dst);

}

void register_batched_mm_functions() {
    add_batched_mm_function(&naive_batched_mm, naive_batched_mm_descr);
    add_batched_mm_function(&batched_mm, batched_mm_descr);
    /* ... Register additional test functions here */
}



