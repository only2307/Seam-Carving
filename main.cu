#include <stdio.h>
#include <stdint.h>

#define FILTER_WIDTH 3
__constant__ float dc_filter1[FILTER_WIDTH * FILTER_WIDTH];
__constant__ float dc_filter2[FILTER_WIDTH * FILTER_WIDTH];

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

char *concatStr(const char *s1, const char *s2)
{
    char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

uint8_t min_e_idx(int e1, int e2, int e3)
{
    if (e2 <= e1 && e2 <= e3)
    {
        return 1;
    }
    else if (e3 <= e1 && e3 <= e2)
    {
        return 2;
    }
    else if (e1 <= e2 && e1 <= e3)
    {
        return 0;
    }
    return 255;
}

void readPnm(char *fileName, int &numChannels, int &width, int &height, uint8_t *&pixels)
{
    FILE *f = fopen(fileName, "r");
    if (f == NULL)
    {
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    char type[3];
    fscanf(f, "%s", type);
    if (strcmp(type, "P2") == 0)
        numChannels = 1;
    else if (strcmp(type, "P3") == 0)
        numChannels = 3;
    else // In this exercise, we don't touch other types
    {
        fclose(f);
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fscanf(f, "%i", &width);
    fscanf(f, "%i", &height);

    int max_val;
    fscanf(f, "%i", &max_val);
    if (max_val > 255) // In this exercise, we assume 1 byte per value
    {
        fclose(f);
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    pixels = (uint8_t *)malloc(width * height * numChannels * sizeof(uint8_t));
    for (int i = 0; i < width * height * numChannels; i++)
        fscanf(f, "%hhu", &pixels[i]);

    fclose(f);
}

void writePnm(uint8_t *pixels, int numChannels, int width, int height, char *fileName)
{
    FILE *f = fopen(fileName, "w");
    if (f == NULL)
    {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    if (numChannels == 1)
        fprintf(f, "P2\n%i %i\n255\n", width, height);
    else if (numChannels == 3)
        fprintf(f, "P3\n%i %i\n255\n", width, height);
    else
    {
        fclose(f);
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }
    for (int h = 0; h < height * numChannels; h++)
    {
        for (int w = 0; w < width; w++)
        {
            fprintf(f, "%hhu ", pixels[h * width + w]);
            if (w == width - 1)
            {
                fprintf(f, "\n");
            }
        }
    }
    fclose(f);
}

// ======================================== KERNELs ========================================
__global__ void grayscaleOnDevice(int width, int height, uint8_t *inPixels, uint8_t *grayscalePixels)
{
    // for (int r = 0; r < height; r++)
    // {
    //     for (int c = 0; c < width; c++)
    //     {
    //         int i = r * width + c;
    //         uint8_t red = inPixels[3 * i];
    //         uint8_t green = inPixels[3 * i + 1];
    //         uint8_t blue = inPixels[3 * i + 2];
    //         grayscalePixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
    //     }
    // }
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width)
    {
        int i = r * width + c;
        uint8_t red = inPixels[3 * i];
        uint8_t green = inPixels[3 * i + 1];
        uint8_t blue = inPixels[3 * i + 2];
        grayscalePixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
    }
}

__global__ void convolutionOnDevice(int width, int height, int filterWidth, uint8_t *grayscalePixels, float *filter1, float *filter2, uint8_t *convoPixels)
{
    // for (int r = 0; r < height; r++)
    // {
    //     for (int c = 0; c < width; c++)
    //     {
    //         float new_pixel_1 = 0;
    //         float new_pixel_2 = 0;
    //         for (int j = 0; j < filterWidth; j++)
    //         {
    //             for (int k = 0; k < filterWidth; k++)
    //             {
    //                 int current_row = r + j - (filterWidth / 2);
    //                 int current_column = c + k - (filterWidth / 2);
    //                 if (current_row < 0)
    //                 {
    //                     current_row = 0;
    //                 }
    //                 else if (current_row > height - 1)
    //                 {
    //                     current_row = height - 1;
    //                 }
    //                 if (current_column < 0)
    //                 {
    //                     current_column = 0;
    //                 }
    //                 else if (current_column > width - 1)
    //                 {
    //                     current_column = width - 1;
    //                 }
    //                 int i = current_row * width + current_column;
    //                 new_pixel_1 += grayscalePixels[i] * filter1[j * filterWidth + k];
    //                 new_pixel_2 += grayscalePixels[i] * filter2[j * filterWidth + k];
    //             }
    //         }
    //         float new_pixel = sqrt(pow(new_pixel_1, 2) + pow(new_pixel_2, 2)); // sqrt(sobel_x ^ 2 + sobel_y ^ 2)
    //         if (new_pixel > 255)
    //         {
    //             new_pixel = 255;
    //         }
    //         else if (new_pixel < 0)
    //         {
    //             new_pixel = 0;
    //         }
    //         int i = r * width + c;
    //         convoPixels[i] = new_pixel;
    //     }
    // }

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    // // For dc_filter debug
    // if (r == 0 && c == 0)
    // {
    //     for (int i = 0; i < filterWidth; i++)
    //     {
    //         for (int j = 0; j < filterWidth; j++)
    //         {
    //             printf("%.0f %.0f | ", filter2[i * filterWidth + j], dc_filter2[i * filterWidth + j]);
    //         }
    //     }
    // }

    if (r < height && c < width)
    {
        float new_pixel_1 = 0;
        float new_pixel_2 = 0;
        for (int j = 0; j < filterWidth; j++)
        {
            for (int k = 0; k < filterWidth; k++)
            {
                int current_row = r + j - (filterWidth / 2);
                int current_column = c + k - (filterWidth / 2);
                if (current_row < 0)
                {
                    current_row = 0;
                }
                else if (current_row > height - 1)
                {
                    current_row = height - 1;
                }
                if (current_column < 0)
                {
                    current_column = 0;
                }
                else if (current_column > width - 1)
                {
                    current_column = width - 1;
                }
                int i = current_row * width + current_column;
                new_pixel_1 += grayscalePixels[i] * filter1[j * filterWidth + k];
                new_pixel_2 += grayscalePixels[i] * filter2[j * filterWidth + k];
            }
        }
        float new_pixel = sqrt(pow(new_pixel_1, 2) + pow(new_pixel_2, 2)); // sqrt(sobel_x ^ 2 + sobel_y ^ 2)
        if (new_pixel > 255)
        {
            new_pixel = 255;
        }
        else if (new_pixel < 0)
        {
            new_pixel = 0;
        }
        int i = r * width + c;
        convoPixels[i] = new_pixel;
    }
}

__global__ void convolutionOnDeviceOpt1(int width, int height, int filterWidth, uint8_t *grayscalePixels, uint8_t *convoPixels)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width)
    {
        float new_pixel_1 = 0;
        float new_pixel_2 = 0;
        for (int j = 0; j < filterWidth; j++)
        {
            for (int k = 0; k < filterWidth; k++)
            {
                int current_row = r + j - (filterWidth / 2);
                int current_column = c + k - (filterWidth / 2);
                if (current_row < 0)
                {
                    current_row = 0;
                }
                else if (current_row > height - 1)
                {
                    current_row = height - 1;
                }
                if (current_column < 0)
                {
                    current_column = 0;
                }
                else if (current_column > width - 1)
                {
                    current_column = width - 1;
                }
                int i = current_row * width + current_column;
                new_pixel_1 += grayscalePixels[i] * dc_filter1[j * filterWidth + k];
                new_pixel_2 += grayscalePixels[i] * dc_filter2[j * filterWidth + k];
            }
        }
        float new_pixel = sqrt(pow(new_pixel_1, 2) + pow(new_pixel_2, 2)); // sqrt(sobel_x ^ 2 + sobel_y ^ 2)
        if (new_pixel > 255)
        {
            new_pixel = 255;
        }
        else if (new_pixel < 0)
        {
            new_pixel = 0;
        }
        int i = r * width + c;
        convoPixels[i] = new_pixel;
    }
}

__global__ void minEnergiesOnDevice(int width, int height, uint8_t *convoPixels, uint8_t *backtrack, uint8_t *minEnergies)
{
    // // The remained above rows
    // for (int r = 1; r < height; r++)
    // {
    //     for (int c = 0; c < width; c++)
    //     {
    //         int i = (height - 1 - r) * width + c;
    //         int e[3] = {99999, 99999, 99999};
    //         if (c == 0)
    //         {
    //             e[1] = minEnergies[((height - 1 - r) + 1) * width + c];
    //             e[2] = minEnergies[((height - 1 - r) + 1) * width + (c + 1)];
    //         }
    //         else if (c == width - 1)
    //         {
    //             e[0] = minEnergies[((height - 1 - r) + 1) * width + (c - 1)];
    //             e[1] = minEnergies[((height - 1 - r) + 1) * width + c];
    //         }
    //         else
    //         {
    //             e[0] = minEnergies[((height - 1 - r) + 1) * width + (c - 1)];
    //             e[1] = minEnergies[((height - 1 - r) + 1) * width + c];
    //             e[2] = minEnergies[((height - 1 - r) + 1) * width + (c + 1)];
    //         }
    //         // uint8_t min_idx = min_e_idx(e[0], e[1], e[2]);
    //         uint8_t min_idx = 255;
    //         if (e[1] <= e[0] && e[1] <= e[2])
    //         {
    //             min_idx = 1; // return 1;
    //         }
    //         else if (e[2] <= e[0] && e[2] <= e[1])
    //         {
    //             min_idx = 2; // return 2;
    //         }
    //         else if (e[0] <= e[1] && e[0] <= e[2])
    //         {
    //             min_idx = 0; // return 0;
    //         }
    //         int tempMinE = convoPixels[i] + e[min_idx]; // Calculate 1 minimal energy base on 3-energies-below
    //         backtrack[i] = min_idx;                     // Save the direction (which of the 3-energies-below)
    //         if (tempMinE > 255)
    //         {
    //             tempMinE = 255;
    //         }
    //         else if (tempMinE < 0)
    //         {
    //             tempMinE = 0;
    //         }
    //         minEnergies[i] = tempMinE * 0.9; // Special
    //     }
    // }

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width)
    {
        // The remained above rows
        if (r != 0)
        {
        }
    }
}

__global__ void vMinEnergiesOnDevice(int width, int height, uint8_t *convoPixels, int *vMinEnergies)
{

    // int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < width)
    {
        int temp_sum = 0;
        for (int r = 0; r < height; r++)
        {
            int i = r * width + c;
            temp_sum += convoPixels[i];
        }
        vMinEnergies[c] = temp_sum;
    }
}

__global__ void minSeamBacktrackOnDevice(int width, int height, uint8_t *backtrack, int *min_seam_track)
{
    // for (int r = 1; r < height; r++)
    // {
    //     int prev_c = min_seam_track[r - 1];
    //     uint8_t direction = backtrack[(r - 1) * width + prev_c];
    //     if (direction == 0)
    //     {
    //         if (prev_c == 0)
    //         {
    //             min_seam_track[r] = 0;
    //         }
    //         else
    //         {
    //             min_seam_track[r] = prev_c - 1;
    //         }
    //     }
    //     else if (direction == 1)
    //     {
    //         min_seam_track[r] = prev_c;
    //     }
    //     else if (direction == 2)
    //     {
    //         if (prev_c == width - 1)
    //         {
    //             min_seam_track[r] = width - 1;
    //         }
    //         else
    //         {
    //             min_seam_track[r] = prev_c + 1;
    //         }
    //     }
    // }

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width)
    {
    }
}

__global__ void removeMinSeamOnDevice(int width, int height, uint8_t *inPixels, int *min_seam_track, uint8_t *outPixels)
{
    // for (int r = 0; r < height; r++)
    // {
    //     int ignored_idx = min_seam_track[r];
    //     bool meet_ignored_pixel_yet = false;
    //     for (int c = 0; c < width; c++)
    //     {
    //         int i = r * width + c;
    //         if (meet_ignored_pixel_yet == false)
    //         {
    //             if (c != ignored_idx)
    //             {
    //                 outPixels[i * 3 + 0 - 3 * r] = inPixels[i * 3 + 0];
    //                 outPixels[i * 3 + 1 - 3 * r] = inPixels[i * 3 + 1];
    //                 outPixels[i * 3 + 2 - 3 * r] = inPixels[i * 3 + 2];
    //             }
    //             else
    //             {
    //                 meet_ignored_pixel_yet = true;
    //             }
    //         }
    //         else
    //         {
    //             outPixels[i * 3 + 0 - 3 * r - 3] = inPixels[i * 3 + 0];
    //             outPixels[i * 3 + 1 - 3 * r - 3] = inPixels[i * 3 + 1];
    //             outPixels[i * 3 + 2 - 3 * r - 3] = inPixels[i * 3 + 2];
    //         }
    //     }
    // }

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    // int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height)
    {
        int ignored_idx = min_seam_track[r];
        bool meet_ignored_pixel_yet = false;
        for (int c = 0; c < width; c++)
        {
            int i = r * width + c;
            if (meet_ignored_pixel_yet == false)
            {
                if (c != ignored_idx)
                {
                    outPixels[i * 3 + 0 - 3 * r] = inPixels[i * 3 + 0];
                    outPixels[i * 3 + 1 - 3 * r] = inPixels[i * 3 + 1];
                    outPixels[i * 3 + 2 - 3 * r] = inPixels[i * 3 + 2];
                }
                else
                {
                    meet_ignored_pixel_yet = true;
                }
            }
            else
            {
                outPixels[i * 3 + 0 - 3 * r - 3] = inPixels[i * 3 + 0];
                outPixels[i * 3 + 1 - 3 * r - 3] = inPixels[i * 3 + 1];
                outPixels[i * 3 + 2 - 3 * r - 3] = inPixels[i * 3 + 2];
            }
        }
    }
}

// ======================================== HOST ========================================
void seamCarvingOnHost(uint8_t *inPixels, char *fileName, int numChannels, int width, int height, float *filter1, float *filter2, int filterWidth, dim3 blockSize, uint8_t *grayscalePixels, uint8_t *convoPixels, uint8_t *minEnergies, uint8_t *backtrack, int *vMinEnergies, int *min_seam_track, uint8_t *outPixels, int *minEnergiesInt)
{
    // ============================== Grayscale ==============================

    // uint8_t *grayscalePixels = (uint8_t *)malloc(width * height * 1);
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int i = r * width + c;
            uint8_t red = inPixels[3 * i];
            uint8_t green = inPixels[3 * i + 1];
            uint8_t blue = inPixels[3 * i + 2];
            grayscalePixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
        }
    }
    writePnm(grayscalePixels, 1, width, height, concatStr("out_gray", ".pnm")); // Just to test by eyes

    // ============================== Convolution ==============================

    // uint8_t *convoPixels = (uint8_t *)malloc(width * height * 1);
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            float new_pixel_1 = 0;
            float new_pixel_2 = 0;
            for (int j = 0; j < filterWidth; j++)
            {
                for (int k = 0; k < filterWidth; k++)
                {
                    int current_row = r + j - (filterWidth / 2);
                    int current_column = c + k - (filterWidth / 2);
                    if (current_row < 0)
                    {
                        current_row = 0;
                    }
                    else if (current_row > height - 1)
                    {
                        current_row = height - 1;
                    }
                    if (current_column < 0)
                    {
                        current_column = 0;
                    }
                    else if (current_column > width - 1)
                    {
                        current_column = width - 1;
                    }
                    int i = current_row * width + current_column;
                    new_pixel_1 += grayscalePixels[i] * filter1[j * filterWidth + k];
                    new_pixel_2 += grayscalePixels[i] * filter2[j * filterWidth + k];
                }
            }
            float new_pixel = sqrt(pow(new_pixel_1, 2) + pow(new_pixel_2, 2)); // sqrt(sobel_x ^ 2 + sobel_y ^ 2)
            if (new_pixel > 255)
            {
                new_pixel = 255;
            }
            else if (new_pixel < 0)
            {
                new_pixel = 0;
            }
            int i = r * width + c;
            convoPixels[i] = new_pixel;
        }
    }
    writePnm(convoPixels, 1, width, height, concatStr("out_edge", ".pnm")); // Just to test by eyes

    // ============================== Energy to Min Energy ==============================

    // Loop from bottom to top row
    // -> Calculate the min energy of each pixel:
    //    +) The bottom row: Just copy the energy -> min energy
    //    +) The remained above rows: Calculate min energy on the current row by using:
    //                                -) The current energy
    //                                -) The min of 3-min-energy-on-the-row-below
    // Simultaneously backtrack the direction of each pixel (which of the 3-energies-below)

    // uint8_t *minEnergies = (uint8_t *)malloc(width * height * 1);
    // uint8_t *backtrack = (uint8_t *)malloc(width * height * 1);
    for (int r = height - 1; r >= 0; r--)
    {
        for (int c = 0; c < width; c++)
        {
            int i = r * width + c;
            int tempMinE;

            // The bottom row
            if (r == height - 1)
            {
                tempMinE = convoPixels[i];
                backtrack[i] = 1;
            }
            // The remained above rows
            else
            {
                int e[3] = {99999, 99999, 99999};
                if (c == 0)
                {
                    e[1] = minEnergiesInt[(r + 1) * width + c];
                    e[2] = minEnergiesInt[(r + 1) * width + (c + 1)];
                }
                else if (c == width - 1)
                {
                    e[0] = minEnergiesInt[(r + 1) * width + (c - 1)];
                    e[1] = minEnergiesInt[(r + 1) * width + c];
                }
                else
                {
                    e[0] = minEnergiesInt[(r + 1) * width + (c - 1)];
                    e[1] = minEnergiesInt[(r + 1) * width + c];
                    e[2] = minEnergiesInt[(r + 1) * width + (c + 1)];
                }
                uint8_t min_idx = min_e_idx(e[0], e[1], e[2]);

                tempMinE = convoPixels[i] + e[min_idx]; // Calculate 1 minimal energy base on 3-energies-below
                backtrack[i] = min_idx;                 // Save the direction (which of the 3-energies-below)
            }

            // if (tempMinE > 255)
            // {
            //     tempMinE = 255;
            // }
            // else if (tempMinE < 0)
            // {
            //     tempMinE = 0;
            // }
            // minEnergies[i] = tempMinE * 0.9; // Special

            minEnergiesInt[i] = tempMinE * 1.0;
        }
    }
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int i = r * width + c;
            int tempMinE = 0;
            tempMinE = minEnergiesInt[i];
            minEnergies[i] = tempMinE * 0.02;
        }
    }
    writePnm(minEnergies, 1, width, height, concatStr("out_minEnergy", ".pnm")); // Just to test by eyes

    // ============================== Extra: Vertical Min Energy ==============================

    // Loop through columns and calculate the sum of energies of each column
    // This will be used to help to choose which pixel to start when find the min seam later

    // int *vMinEnergies = (int *)malloc(width);
    for (int c = 0; c < width; c++)
    {
        int temp_sum = 0;
        for (int r = 0; r < height; r++)
        {
            int i = r * width + c;
            temp_sum += convoPixels[i];
        }
        vMinEnergies[c] = temp_sum;
    }

    // ============================== Find the Min Seam ==============================

    // First: Find the best pixel (on the top row) to start.
    //        This pixel will be used to start the backtrack.
    // Finding that starting pixel will depend on:
    // +) The min energy of that pixel (of course, the min value on the row)
    // +) If there are multiple pixels have same min value
    //    -> Use vMinEnergies calculated above

    // int *min_seam_track = (int *)malloc(height);
    for (int c = 0; c < width; c++)
    {
        uint8_t temp_min_e;
        int temp_min_v_e;
        if (c == 0)
        {
            temp_min_e = minEnergies[0];
            temp_min_v_e = vMinEnergies[0];
            min_seam_track[0] = 0;
        }
        else
        {
            bool is_satisfied = false;
            // ----- Just a trick -----
            if (width % 2 == 0)
            {
                if (minEnergies[c] < temp_min_e && vMinEnergies[c] <= temp_min_v_e)
                {
                    is_satisfied = true;
                }
            }
            else
            {
                if (minEnergies[c] <= temp_min_e && vMinEnergies[c] < temp_min_v_e)
                {
                    is_satisfied = true;
                }
            }
            if (is_satisfied == true)
            {
                temp_min_e = minEnergies[c];
                temp_min_v_e = vMinEnergies[c];
                min_seam_track[0] = c;
            }
        }
    }

    // Second: With the starting pixel:
    //         Backtrack from row to row to find a list of pixels
    //         -> The seam with min energy

    for (int r = 1; r < height; r++)
    {
        int prev_c = min_seam_track[r - 1];
        uint8_t direction = backtrack[(r - 1) * width + prev_c];
        if (direction == 0)
        {
            if (prev_c == 0)
            {
                min_seam_track[r] = 0;
            }
            else
            {
                min_seam_track[r] = prev_c - 1;
            }
        }
        else if (direction == 1)
        {
            min_seam_track[r] = prev_c;
        }
        else if (direction == 2)
        {
            if (prev_c == width - 1)
            {
                min_seam_track[r] = width - 1;
            }
            else
            {
                min_seam_track[r] = prev_c + 1;
            }
        }
    }

    // ============================== Remove Min Seam from image ==============================

    // Copy all pixels from inPixels to outPixels, EXCEPT the min seam
    // Just it!

    // uint8_t *outPixels = (uint8_t *)malloc((width - 1) * height * numChannels * sizeof(uint8_t)); // width-1: means that the output image will be cut 1 pixel (the min seam)
    for (int r = 0; r < height; r++)
    {
        int ignored_idx = min_seam_track[r];
        bool meet_ignored_pixel_yet = false;
        for (int c = 0; c < width; c++)
        {
            int i = r * width + c;
            if (meet_ignored_pixel_yet == false)
            {
                if (c != ignored_idx)
                {
                    outPixels[i * 3 + 0 - 3 * r] = inPixels[i * 3 + 0];
                    outPixels[i * 3 + 1 - 3 * r] = inPixels[i * 3 + 1];
                    outPixels[i * 3 + 2 - 3 * r] = inPixels[i * 3 + 2];
                }
                else
                {
                    meet_ignored_pixel_yet = true;
                }
            }
            else
            {
                outPixels[i * 3 + 0 - 3 * r - 3] = inPixels[i * 3 + 0];
                outPixels[i * 3 + 1 - 3 * r - 3] = inPixels[i * 3 + 1];
                outPixels[i * 3 + 2 - 3 * r - 3] = inPixels[i * 3 + 2];
            }
        }
    }
    writePnm(outPixels, 3, width - 1, height, fileName); // Just to test by eyes
    // ============================== =========================== ==============================
}

// ======================================== DEVICE ========================================
void seamCarvingOnDevice(uint8_t *inPixels, char *fileName, int numChannels, int width, int height, float *filter1, float *filter2, int filterWidth, dim3 blockSize, uint8_t *grayscalePixels, uint8_t *convoPixels, uint8_t *minEnergies, uint8_t *backtrack, int *vMinEnergies, int *min_seam_track, uint8_t *outPixels, uint8_t *d_inPixels, size_t nBytes_inPixels, uint8_t *d_grayscalePixels, size_t nBytes_grayscalePixels, uint8_t *d_convoPixels, size_t nBytes_convoPixels, float *d_filter1, float *d_filter2, size_t nBytes_filter, uint8_t *d_backtrack, size_t nBytes_backtrack, uint8_t *d_minEnergies, size_t nBytes_minEnergies, int *d_vMinEnergies, size_t nBytes_vMinEnergies, int *d_min_seam_track, size_t nBytes_min_seam_track, uint8_t *d_outPixels, size_t nBytes_outPixels, int *minEnergiesInt)
{
    // Gridsize
    dim3 gridSize(
        (width - 1) / blockSize.x + 1,
        (height - 1) / blockSize.y + 1);

    // ============================== Grayscale (Parallel) ==============================

    // uint8_t *grayscalePixels = (uint8_t *)malloc(width * height * 1);
    // for (int r = 0; r < height; r++)
    // {
    //     for (int c = 0; c < width; c++)
    //     {
    //         int i = r * width + c;
    //         uint8_t red = inPixels[3 * i];
    //         uint8_t green = inPixels[3 * i + 1];
    //         uint8_t blue = inPixels[3 * i + 2];
    //         grayscalePixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
    //     }
    // }

    // Copy data
    CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes_inPixels, cudaMemcpyHostToDevice));
    // Call kernel
    grayscaleOnDevice<<<gridSize, blockSize>>>(width, height, d_inPixels, d_grayscalePixels);
    CHECK(cudaDeviceSynchronize());
    // Copy data
    CHECK(cudaMemcpy(grayscalePixels, d_grayscalePixels, nBytes_grayscalePixels, cudaMemcpyDeviceToHost));

    // Output
    writePnm(grayscalePixels, 1, width, height, concatStr("out_gray", ".pnm"));

    // ============================== Convolution (Parallel) ==============================

    // uint8_t *convoPixels = (uint8_t *)malloc(width * height * 1);
    // for (int r = 0; r < height; r++)
    // {
    //     for (int c = 0; c < width; c++)
    //     {
    //         float new_pixel_1 = 0;
    //         float new_pixel_2 = 0;
    //         for (int j = 0; j < filterWidth; j++)
    //         {
    //             for (int k = 0; k < filterWidth; k++)
    //             {
    //                 int current_row = r + j - (filterWidth / 2);
    //                 int current_column = c + k - (filterWidth / 2);
    //                 if (current_row < 0)
    //                 {
    //                     current_row = 0;
    //                 }
    //                 else if (current_row > height - 1)
    //                 {
    //                     current_row = height - 1;
    //                 }
    //                 if (current_column < 0)
    //                 {
    //                     current_column = 0;
    //                 }
    //                 else if (current_column > width - 1)
    //                 {
    //                     current_column = width - 1;
    //                 }
    //                 int i = current_row * width + current_column;
    //                 new_pixel_1 += grayscalePixels[i] * filter1[j * filterWidth + k];
    //                 new_pixel_2 += grayscalePixels[i] * filter2[j * filterWidth + k];
    //             }
    //         }
    //         float new_pixel = sqrt(pow(new_pixel_1, 2) + pow(new_pixel_2, 2)); // sqrt(sobel_x ^ 2 + sobel_y ^ 2)
    //         if (new_pixel > 255)
    //         {
    //             new_pixel = 255;
    //         }
    //         else if (new_pixel < 0)
    //         {
    //             new_pixel = 0;
    //         }
    //         int i = r * width + c;
    //         convoPixels[i] = new_pixel;
    //     }
    // }

    // Copy data
    CHECK(cudaMemcpy(d_grayscalePixels, grayscalePixels, nBytes_grayscalePixels, cudaMemcpyHostToDevice)); // No need
    CHECK(cudaMemcpy(d_filter1, filter1, nBytes_filter, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filter2, filter2, nBytes_filter, cudaMemcpyHostToDevice));
    // Call kernel
    convolutionOnDevice<<<gridSize, blockSize>>>(width, height, filterWidth, d_grayscalePixels, d_filter1, d_filter2, d_convoPixels);
    CHECK(cudaDeviceSynchronize());
    // Copy data
    CHECK(cudaMemcpy(convoPixels, d_convoPixels, nBytes_convoPixels, cudaMemcpyDeviceToHost));

    // Output
    writePnm(convoPixels, 1, width, height, concatStr("out_edge", ".pnm"));

    // ============================== Energy to Min Energy ==============================

    // Loop from bottom to top row
    // -> Calculate the min energy of each pixel:
    //    +) The bottom row: Just copy the energy -> min energy
    //    +) The remained above rows: Calculate min energy on the current row by using:
    //                                -) The current energy
    //                                -) The min of 3-min-energy-on-the-row-below
    // Simultaneously backtrack the direction of each pixel (which of the 3-energies-below)

    // uint8_t *minEnergies = (uint8_t *)malloc(width * height * 1);
    // uint8_t *backtrack = (uint8_t *)malloc(width * height * 1);

    // // The bottom row
    for (int c = 0; c < width; c++)
    {
        int i = (height - 1) * width + c;
        int tempMinE = convoPixels[i];

        // if (tempMinE > 255)
        // {
        //     tempMinE = 255;
        // }
        // else if (tempMinE < 0)
        // {
        //     tempMinE = 0;
        // }
        // minEnergies[i] = tempMinE * 0.9; // Special

        minEnergiesInt[i] = tempMinE * 1.0;

        backtrack[i] = 1;
    }

    // // The remained above rows
    for (int r = 1; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int i = (height - 1 - r) * width + c;
            int e[3] = {99999, 99999, 99999};
            if (c == 0)
            {
                e[1] = minEnergiesInt[((height - 1 - r) + 1) * width + c];
                e[2] = minEnergiesInt[((height - 1 - r) + 1) * width + (c + 1)];
            }
            else if (c == width - 1)
            {
                e[0] = minEnergiesInt[((height - 1 - r) + 1) * width + (c - 1)];
                e[1] = minEnergiesInt[((height - 1 - r) + 1) * width + c];
            }
            else
            {
                e[0] = minEnergiesInt[((height - 1 - r) + 1) * width + (c - 1)];
                e[1] = minEnergiesInt[((height - 1 - r) + 1) * width + c];
                e[2] = minEnergiesInt[((height - 1 - r) + 1) * width + (c + 1)];
            }
            // uint8_t min_idx = min_e_idx(e[0], e[1], e[2]);
            uint8_t min_idx = 255;
            if (e[1] <= e[0] && e[1] <= e[2])
            {
                min_idx = 1; // return 1;
            }
            else if (e[2] <= e[0] && e[2] <= e[1])
            {
                min_idx = 2; // return 2;
            }
            else if (e[0] <= e[1] && e[0] <= e[2])
            {
                min_idx = 0; // return 0;
            }
            int tempMinE = convoPixels[i] + e[min_idx]; // Calculate 1 minimal energy base on 3-energies-below

            // if (tempMinE > 255)
            // {
            //     tempMinE = 255;
            // }
            // else if (tempMinE < 0)
            // {
            //     tempMinE = 0;
            // }
            // minEnergies[i] = tempMinE * 0.9; // Special

            minEnergiesInt[i] = tempMinE * 1.0;

            backtrack[i] = min_idx; // Save the direction (which of the 3-energies-below)
        }
    }

    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int i = r * width + c;
            int tempMinE = 0;
            tempMinE = minEnergiesInt[i];
            minEnergies[i] = tempMinE * 0.02;
        }
    }

    // Output
    writePnm(minEnergies, 1, width, height, concatStr("out_minEnergy", ".pnm"));

    // ============================== Extra: Vertical Min Energy (Parallel) ==============================

    // Loop through columns and calculate the sum of energies of each column
    // This will be used to help to choose which pixel to start when find the min seam later

    // int *vMinEnergies = (int *)malloc(width);
    // for (int c = 0; c < width; c++)
    // {
    //     int temp_sum = 0;
    //     for (int r = 0; r < height; r++)
    //     {
    //         int i = r * width + c;
    //         temp_sum += convoPixels[i];
    //     }
    //     vMinEnergies[c] = temp_sum;
    // }

    // Copy data
    CHECK(cudaMemcpy(d_convoPixels, convoPixels, nBytes_convoPixels, cudaMemcpyHostToDevice)); // No need
    // Call kernel
    vMinEnergiesOnDevice<<<gridSize, blockSize>>>(width, height, d_convoPixels, d_vMinEnergies);
    CHECK(cudaDeviceSynchronize());
    // Copy data
    CHECK(cudaMemcpy(vMinEnergies, d_vMinEnergies, nBytes_vMinEnergies, cudaMemcpyDeviceToHost));

    // ============================== Find the Min Seam ==============================

    // ------------------------------ First step (No parallel) ------------------------------

    // First: Find the best pixel (on the top row) to start.
    //        This pixel will be used to start the backtrack.
    // Finding that starting pixel will depend on:
    // +) The min energy of that pixel (of course, the min value on the row)
    // +) If there are multiple pixels have same min value
    //    -> Use vMinEnergies calculated above

    // int *min_seam_track = (int *)malloc(height);
    for (int c = 0; c < width; c++)
    {
        uint8_t temp_min_e;
        int temp_min_v_e;
        if (c == 0)
        {
            temp_min_e = minEnergies[0];
            temp_min_v_e = vMinEnergies[0];
            min_seam_track[0] = 0;
        }
        else
        {
            bool is_satisfied = false;
            // ----- Just a trick -----
            if (width % 2 == 0)
            {
                if (minEnergies[c] < temp_min_e && vMinEnergies[c] <= temp_min_v_e)
                {
                    is_satisfied = true;
                }
            }
            else
            {
                if (minEnergies[c] <= temp_min_e && vMinEnergies[c] < temp_min_v_e)
                {
                    is_satisfied = true;
                }
            }
            if (is_satisfied == true)
            {
                temp_min_e = minEnergies[c];
                temp_min_v_e = vMinEnergies[c];
                min_seam_track[0] = c;
            }
        }
    }

    // ------------------------------ Second step (No parallel) ------------------------------

    // Second: With the starting pixel:
    //         Backtrack from row to row to find a list of pixels
    //         -> The seam with min energy

    for (int r = 1; r < height; r++)
    {
        int prev_c = min_seam_track[r - 1];
        uint8_t direction = backtrack[(r - 1) * width + prev_c];
        if (direction == 0)
        {
            if (prev_c == 0)
            {
                min_seam_track[r] = 0;
            }
            else
            {
                min_seam_track[r] = prev_c - 1;
            }
        }
        else if (direction == 1)
        {
            min_seam_track[r] = prev_c;
        }
        else if (direction == 2)
        {
            if (prev_c == width - 1)
            {
                min_seam_track[r] = width - 1;
            }
            else
            {
                min_seam_track[r] = prev_c + 1;
            }
        }
    }

    // ============================== Remove Min Seam from image (Parallel) ==============================

    // Copy all pixels from inPixels to outPixels, EXCEPT the min seam
    // Just it!

    // uint8_t *outPixels = (uint8_t *)malloc((width - 1) * height * numChannels * sizeof(uint8_t)); // width-1: means that the output image will be cut 1 pixel (the min seam)
    // for (int r = 0; r < height; r++)
    // {
    //     int ignored_idx = min_seam_track[r];
    //     bool meet_ignored_pixel_yet = false;
    //     for (int c = 0; c < width; c++)
    //     {
    //         int i = r * width + c;
    //         if (meet_ignored_pixel_yet == false)
    //         {
    //             if (c != ignored_idx)
    //             {
    //                 outPixels[i * 3 + 0 - 3 * r] = inPixels[i * 3 + 0];
    //                 outPixels[i * 3 + 1 - 3 * r] = inPixels[i * 3 + 1];
    //                 outPixels[i * 3 + 2 - 3 * r] = inPixels[i * 3 + 2];
    //             }
    //             else
    //             {
    //                 meet_ignored_pixel_yet = true;
    //             }
    //         }
    //         else
    //         {
    //             outPixels[i * 3 + 0 - 3 * r - 3] = inPixels[i * 3 + 0];
    //             outPixels[i * 3 + 1 - 3 * r - 3] = inPixels[i * 3 + 1];
    //             outPixels[i * 3 + 2 - 3 * r - 3] = inPixels[i * 3 + 2];
    //         }
    //     }
    // }

    // Copy data
    CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes_inPixels, cudaMemcpyHostToDevice)); // No need
    CHECK(cudaMemcpy(d_min_seam_track, min_seam_track, nBytes_min_seam_track, cudaMemcpyHostToDevice));
    // Call kernel
    removeMinSeamOnDevice<<<gridSize, blockSize>>>(width, height, d_inPixels, d_min_seam_track, d_outPixels);
    CHECK(cudaDeviceSynchronize());
    // Copy data
    CHECK(cudaMemcpy(outPixels, d_outPixels, nBytes_outPixels, cudaMemcpyDeviceToHost));

    // Output
    writePnm(outPixels, 3, width - 1, height, fileName);
    // ============================== =========================== ==============================
}

// ======================================== DEVICE (OPTIMIZED) ========================================
void seamCarvingOnDeviceOpt1(uint8_t *inPixels, char *fileName, int numChannels, int width, int height, int filterWidth, dim3 blockSize, uint8_t *grayscalePixels, uint8_t *convoPixels, uint8_t *minEnergies, uint8_t *backtrack, int *vMinEnergies, int *min_seam_track, uint8_t *outPixels, uint8_t *d_inPixels, size_t nBytes_inPixels, uint8_t *d_grayscalePixels, size_t nBytes_grayscalePixels, uint8_t *d_convoPixels, size_t nBytes_convoPixels, uint8_t *d_backtrack, size_t nBytes_backtrack, uint8_t *d_minEnergies, size_t nBytes_minEnergies, int *d_vMinEnergies, size_t nBytes_vMinEnergies, int *d_min_seam_track, size_t nBytes_min_seam_track, uint8_t *d_outPixels, size_t nBytes_outPixels, int *minEnergiesInt)
{
    // Gridsize
    dim3 gridSize(
        (width - 1) / blockSize.x + 1,
        (height - 1) / blockSize.y + 1);

    // ============================== Grayscale (Parallel) ==============================

    // Copy data
    CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes_inPixels, cudaMemcpyHostToDevice));
    // Call kernel
    grayscaleOnDevice<<<gridSize, blockSize>>>(width, height, d_inPixels, d_grayscalePixels);
    CHECK(cudaDeviceSynchronize());
    // Copy data
    CHECK(cudaMemcpy(grayscalePixels, d_grayscalePixels, nBytes_grayscalePixels, cudaMemcpyDeviceToHost));

    // // Output
    // writePnm(grayscalePixels, 1, width, height, concatStr("out_gray", ".pnm"));

    // ============================== Convolution (Parallel) ==============================

    // Copy data
    // CHECK(cudaMemcpy(d_grayscalePixels, grayscalePixels, nBytes_grayscalePixels, cudaMemcpyHostToDevice)); // No need
    // CHECK(cudaMemcpy(d_filter1, filter1, nBytes_filter, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_filter2, filter2, nBytes_filter, cudaMemcpyHostToDevice));
    // Call kernel
    convolutionOnDeviceOpt1<<<gridSize, blockSize>>>(width, height, filterWidth, d_grayscalePixels, d_convoPixels);
    CHECK(cudaDeviceSynchronize());
    // Copy data
    CHECK(cudaMemcpy(convoPixels, d_convoPixels, nBytes_convoPixels, cudaMemcpyDeviceToHost));

    // // Output
    // writePnm(convoPixels, 1, width, height, concatStr("out_edge", ".pnm"));

    // ============================== Energy to Min Energy ==============================

    // Loop from bottom to top row
    // -> Calculate the min energy of each pixel:
    //    +) The bottom row: Just copy the energy -> min energy
    //    +) The remained above rows: Calculate min energy on the current row by using:
    //                                -) The current energy
    //                                -) The min of 3-min-energy-on-the-row-below
    // Simultaneously backtrack the direction of each pixel (which of the 3-energies-below)

    // // The bottom row
    for (int c = 0; c < width; c++)
    {
        int i = (height - 1) * width + c;
        int tempMinE = convoPixels[i];
        minEnergiesInt[i] = tempMinE * 1.0;
        backtrack[i] = 1;
    }

    // // The remained above rows
    for (int r = 1; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int i = (height - 1 - r) * width + c;
            int e[3] = {99999, 99999, 99999};
            if (c == 0)
            {
                e[1] = minEnergiesInt[((height - 1 - r) + 1) * width + c];
                e[2] = minEnergiesInt[((height - 1 - r) + 1) * width + (c + 1)];
            }
            else if (c == width - 1)
            {
                e[0] = minEnergiesInt[((height - 1 - r) + 1) * width + (c - 1)];
                e[1] = minEnergiesInt[((height - 1 - r) + 1) * width + c];
            }
            else
            {
                e[0] = minEnergiesInt[((height - 1 - r) + 1) * width + (c - 1)];
                e[1] = minEnergiesInt[((height - 1 - r) + 1) * width + c];
                e[2] = minEnergiesInt[((height - 1 - r) + 1) * width + (c + 1)];
            }
            // uint8_t min_idx = min_e_idx(e[0], e[1], e[2]);
            uint8_t min_idx = 255;
            if (e[1] <= e[0] && e[1] <= e[2])
            {
                min_idx = 1; // return 1;
            }
            else if (e[2] <= e[0] && e[2] <= e[1])
            {
                min_idx = 2; // return 2;
            }
            else if (e[0] <= e[1] && e[0] <= e[2])
            {
                min_idx = 0; // return 0;
            }
            int tempMinE = convoPixels[i] + e[min_idx]; // Calculate 1 minimal energy base on 3-energies-below
            minEnergiesInt[i] = tempMinE * 1.0;
            backtrack[i] = min_idx; // Save the direction (which of the 3-energies-below)
        }
    }

    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int i = r * width + c;
            int tempMinE = 0;
            tempMinE = minEnergiesInt[i];
            minEnergies[i] = tempMinE * 0.02;
        }
    }

    // // Output
    // writePnm(minEnergies, 1, width, height, concatStr("out_minEnergy", ".pnm"));
    // ============================== Extra: Vertical Min Energy (Parallel) ==============================

    // Loop through columns and calculate the sum of energies of each column
    // This will be used to help to choose which pixel to start when find the min seam later

    // Copy data
    // CHECK(cudaMemcpy(d_convoPixels, convoPixels, nBytes_convoPixels, cudaMemcpyHostToDevice)); // No need
    // Call kernel
    vMinEnergiesOnDevice<<<gridSize, blockSize>>>(width, height, d_convoPixels, d_vMinEnergies);
    CHECK(cudaDeviceSynchronize());
    // Copy data
    CHECK(cudaMemcpy(vMinEnergies, d_vMinEnergies, nBytes_vMinEnergies, cudaMemcpyDeviceToHost));

    // ============================== Find the Min Seam ==============================

    // ------------------------------ First step (No parallel) ------------------------------

    // First: Find the best pixel (on the top row) to start.
    //        This pixel will be used to start the backtrack.
    // Finding that starting pixel will depend on:
    // +) The min energy of that pixel (of course, the min value on the row)
    // +) If there are multiple pixels have same min value
    //    -> Use vMinEnergies calculated above

    for (int c = 0; c < width; c++)
    {
        uint8_t temp_min_e;
        int temp_min_v_e;
        if (c == 0)
        {
            temp_min_e = minEnergies[0];
            temp_min_v_e = vMinEnergies[0];
            min_seam_track[0] = 0;
        }
        else
        {
            bool is_satisfied = false;
            // ----- Just a trick -----
            if (width % 2 == 0)
            {
                if (minEnergies[c] < temp_min_e && vMinEnergies[c] <= temp_min_v_e)
                {
                    is_satisfied = true;
                }
            }
            else
            {
                if (minEnergies[c] <= temp_min_e && vMinEnergies[c] < temp_min_v_e)
                {
                    is_satisfied = true;
                }
            }
            if (is_satisfied == true)
            {
                temp_min_e = minEnergies[c];
                temp_min_v_e = vMinEnergies[c];
                min_seam_track[0] = c;
            }
        }
    }

    // ------------------------------ Second step (No parallel) ------------------------------

    // Second: With the starting pixel:
    //         Backtrack from row to row to find a list of pixels
    //         -> The seam with min energy

    for (int r = 1; r < height; r++)
    {
        int prev_c = min_seam_track[r - 1];
        uint8_t direction = backtrack[(r - 1) * width + prev_c];
        if (direction == 0)
        {
            if (prev_c == 0)
            {
                min_seam_track[r] = 0;
            }
            else
            {
                min_seam_track[r] = prev_c - 1;
            }
        }
        else if (direction == 1)
        {
            min_seam_track[r] = prev_c;
        }
        else if (direction == 2)
        {
            if (prev_c == width - 1)
            {
                min_seam_track[r] = width - 1;
            }
            else
            {
                min_seam_track[r] = prev_c + 1;
            }
        }
    }

    // ============================== Remove Min Seam from image (Parallel) ==============================

    // Copy all pixels from inPixels to outPixels, EXCEPT the min seam
    // Just it!

    // Copy data
    // CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes_inPixels, cudaMemcpyHostToDevice)); // No need
    CHECK(cudaMemcpy(d_min_seam_track, min_seam_track, nBytes_min_seam_track, cudaMemcpyHostToDevice));
    // Call kernel
    removeMinSeamOnDevice<<<gridSize, blockSize>>>(width, height, d_inPixels, d_min_seam_track, d_outPixels);
    CHECK(cudaDeviceSynchronize());
    // Copy data
    CHECK(cudaMemcpy(outPixels, d_outPixels, nBytes_outPixels, cudaMemcpyDeviceToHost));

    // Output
    writePnm(outPixels, 3, width - 1, height, fileName);
    // ============================== =========================== ==============================
}

// -- DEVICE OPTOMIZE 2 -- //
__global__ void convolutionOnDeviceOpt2(int width, int padding, int rows, int height, int filterWidth, uint8_t *grayscalePixels, float *filter1, float *filter2, uint8_t *convoPixels)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (padding != 0 && r < 1)
        return;
    if (r > rows)
        return;

    if (r < height && c < width)
    {   
        // printf("%d ", r);
        float new_pixel_1 = 0;
        float new_pixel_2 = 0;
        for (int j = 0; j < filterWidth; j++)
        {
            for (int k = 0; k < filterWidth; k++)
            {
                int current_row = r + j - (filterWidth / 2);
                int current_column = c + k - (filterWidth / 2);
                if (current_row < 0)
                {
                    current_row = 0;
                }
                else if (current_row > height - 1)
                {
                    current_row = height - 1;
                }
                if (current_column < 0)
                {
                    current_column = 0;
                }
                else if (current_column > width - 1)
                {
                    current_column = width - 1;
                }
                int i = current_row * width + current_column;
                new_pixel_1 += grayscalePixels[i] * filter1[j * filterWidth + k];
                new_pixel_2 += grayscalePixels[i] * filter2[j * filterWidth + k];
            }
        }
        float new_pixel = sqrt(pow(new_pixel_1, 2) + pow(new_pixel_2, 2)); // sqrt(sobel_x ^ 2 + sobel_y ^ 2)
        if (new_pixel > 255)
        {
            new_pixel = 255;
        }
        else if (new_pixel < 0)
        {
            new_pixel = 0;
        }
        int i = r * width + c;
        convoPixels[i] = new_pixel;
    }
}

void seamCarvingOnDeviceOpt2(uint8_t *inPixels, char *fileName, int numChannels, int width, int height, int nStreams, float *filter1, float *filter2, int filterWidth, dim3 blockSize, uint8_t *grayscalePixels, uint8_t *convoPixels, uint8_t *minEnergies, uint8_t *backtrack, int *vMinEnergies, int *min_seam_track, uint8_t *outPixels, int* minEnergiesInt)
{
    float * d_filter1, *d_filter2;
    uint8_t * d_in, *d_grayscale, *d_convo, *d_backtrack, *d_minE, *d_out;
    int *d_vMinE;
    int * d_min_seam_track;
    // Gridsize
    dim3 gridSize(
        (width - 1) / blockSize.x + 1,
        (height - 1) / blockSize.y + 1);

    // , *d_backtrack, *d_minE, *d_vMinE, *d_minS, *d_out;
    size_t byteColor = width*height*numChannels*sizeof(uint8_t);
    size_t byte = width * height *sizeof(uint8_t);
    size_t byteOut = (width-1) * height * numChannels * sizeof(uint8_t);
    size_t byteSeam = height *sizeof(int);
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    CHECK(cudaMalloc(&d_filter1, filterSize));
    CHECK(cudaMalloc(&d_filter2, filterSize));
    CHECK(cudaMemcpy(d_filter1, filter1, filterSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filter2, filter2, filterSize, cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&d_in, byteColor));
    CHECK(cudaMalloc(&d_grayscale,byte));
    CHECK(cudaMalloc(&d_convo,byte));
    CHECK(cudaMalloc(&d_minE,byte));
    CHECK(cudaMalloc(&d_backtrack,byte));
    CHECK(cudaMalloc(&d_min_seam_track,byteSeam));
    CHECK(cudaMalloc(&d_out,byteOut));
    CHECK(cudaMalloc(&d_vMinE,width*sizeof(int)));
    CHECK(cudaMemcpy(d_vMinE, vMinEnergies, width*sizeof(int), cudaMemcpyHostToDevice));

    int most_rows, last_rows;
    most_rows = (height-1)/nStreams+1;
    last_rows = height - (nStreams-1)*most_rows;
    cudaStream_t *streams = (cudaStream_t *) malloc(nStreams * sizeof(cudaStream_t));
    for (int i = 0;i<nStreams; i++) {
        CHECK(cudaStreamCreate(&streams[i]));
    }
    int done_row = 0;
    for (int i = nStreams - 1; i>=0; i--) {
        int s = nStreams - 1 - i;
        int rows = (i==0?last_rows:most_rows);
        int padding_row = height - done_row - rows;
        int paddingColor = padding_row * width * numChannels;
        size_t bytesColor = rows * width * numChannels * sizeof(uint8_t);
        CHECK(cudaMemcpyAsync(d_in+paddingColor, inPixels+paddingColor, bytesColor, cudaMemcpyHostToDevice, streams[s]));
        dim3 gridGray((width-1)/blockSize.x+1, (rows-1)/blockSize.y+1);
        int padding = padding_row * width;
        size_t bytes = rows * width * sizeof(uint8_t);
        grayscaleOnDevice<<<gridGray, blockSize, 0, streams[s]>>>(width, rows, d_in+paddingColor, d_grayscale+padding);
        CHECK(cudaMemcpyAsync(grayscalePixels+padding, d_grayscale+padding, bytes, cudaMemcpyDeviceToHost, streams[s]));

        int rows_to_conv = rows;
        int h = rows + 2;
        bytes = (rows+1) *width *sizeof(uint8_t);
        if (i==nStreams-1) {
            rows_to_conv--;
            h=rows;
            bytes = rows*width*sizeof(uint8_t);
        }
        if (i==0) {
            rows_to_conv++;
        }
        dim3 gridConv((width-1)/blockSize.x+1, (rows_to_conv-1)/blockSize.y+1);
        convolutionOnDeviceOpt2<<<gridConv, blockSize, 1, streams[s]>>>(width, padding_row, rows_to_conv, h, filterWidth, d_grayscale+padding, d_filter1, d_filter2, d_convo+padding);
        CHECK(cudaMemcpyAsync(convoPixels+padding, d_convo+padding, bytes, cudaMemcpyDeviceToHost, streams[s]));
        done_row+=rows;
    }

    // CHECK(cudaMemcpy(vMinEnergies, d_vMinE, width*sizeof(int), cudaMemcpyDeviceToHost));
    
    // ============================== Energy to Min Energy ==============================

    // Loop from bottom to top row
    // -> Calculate the min energy of each pixel:
    //    +) The bottom row: Just copy the energy -> min energy
    //    +) The remained above rows: Calculate min energy on the current row by using:
    //                                -) The current energy
    //                                -) The min of 3-min-energy-on-the-row-below
    // Simultaneously backtrack the direction of each pixel (which of the 3-energies-below)

    // // The bottom row
    for (int c = 0; c < width; c++)
    {
        int i = (height - 1) * width + c;
        int tempMinE = convoPixels[i];
        minEnergiesInt[i] = tempMinE * 1.0;
        backtrack[i] = 1;
    }

    // // The remained above rows
    for (int r = 1; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int i = (height - 1 - r) * width + c;
            int e[3] = {99999, 99999, 99999};
            if (c == 0)
            {
                e[1] = minEnergiesInt[((height - 1 - r) + 1) * width + c];
                e[2] = minEnergiesInt[((height - 1 - r) + 1) * width + (c + 1)];
            }
            else if (c == width - 1)
            {
                e[0] = minEnergiesInt[((height - 1 - r) + 1) * width + (c - 1)];
                e[1] = minEnergiesInt[((height - 1 - r) + 1) * width + c];
            }
            else
            {
                e[0] = minEnergiesInt[((height - 1 - r) + 1) * width + (c - 1)];
                e[1] = minEnergiesInt[((height - 1 - r) + 1) * width + c];
                e[2] = minEnergiesInt[((height - 1 - r) + 1) * width + (c + 1)];
            }
            // uint8_t min_idx = min_e_idx(e[0], e[1], e[2]);
            uint8_t min_idx = 255;
            if (e[1] <= e[0] && e[1] <= e[2])
            {
                min_idx = 1; // return 1;
            }
            else if (e[2] <= e[0] && e[2] <= e[1])
            {
                min_idx = 2; // return 2;
            }
            else if (e[0] <= e[1] && e[0] <= e[2])
            {
                min_idx = 0; // return 0;
            }
            int tempMinE = convoPixels[i] + e[min_idx]; // Calculate 1 minimal energy base on 3-energies-below
            minEnergiesInt[i] = tempMinE * 1.0;
            backtrack[i] = min_idx; // Save the direction (which of the 3-energies-below)
        }
    }

    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int i = r * width + c;
            int tempMinE = 0;
            tempMinE = minEnergiesInt[i];
            minEnergies[i] = tempMinE * 0.02;
        }
    }

    // // Output
    // writePnm(minEnergies, 1, width, height, concatStr("out_minEnergy", ".pnm"));
    // ============================== Extra: Vertical Min Energy (Parallel) ==============================

    // Loop through columns and calculate the sum of energies of each column
    // This will be used to help to choose which pixel to start when find the min seam later

    // Copy data
    // CHECK(cudaMemcpy(d_convoPixels, convoPixels, nBytes_convoPixels, cudaMemcpyHostToDevice)); // No need
    // Call kernel
    vMinEnergiesOnDevice<<<gridSize, blockSize>>>(width, height, d_convo, d_vMinE);
    CHECK(cudaDeviceSynchronize());
    // Copy data
    CHECK(cudaMemcpy(vMinEnergies, d_vMinE, width * sizeof(int), cudaMemcpyDeviceToHost));
    // ============================== Find the Min Seam ==============================
    for (int c = 0; c < width; c++)
    {
        uint8_t temp_min_e;
        int temp_min_v_e;
        if (c == 0)
        {
            temp_min_e = minEnergies[0];
            temp_min_v_e = vMinEnergies[0];
            min_seam_track[0] = 0;
        }
        else
        {
            bool is_satisfied = false;
            // ----- Just a trick -----
            if (width % 2 == 0)
            {
                if (minEnergies[c] < temp_min_e && vMinEnergies[c] <= temp_min_v_e)
                {
                    is_satisfied = true;
                }
            }
            else
            {
                if (minEnergies[c] <= temp_min_e && vMinEnergies[c] < temp_min_v_e)
                {
                    is_satisfied = true;
                }
            }
            if (is_satisfied == true)
            {
                temp_min_e = minEnergies[c];
                temp_min_v_e = vMinEnergies[c];
                min_seam_track[0] = c;
            }
        }
    }

    // ------------------------------ Second step (No parallel) ------------------------------

    for (int r = 1; r < height; r++)
    {
        int prev_c = min_seam_track[r - 1];
        uint8_t direction = backtrack[(r - 1) * width + prev_c];
        if (direction == 0)
        {
            if (prev_c == 0)
            {
                min_seam_track[r] = 0;
            }
            else
            {
                min_seam_track[r] = prev_c - 1;
            }
        }
        else if (direction == 1)
        {
            min_seam_track[r] = prev_c;
        }
        else if (direction == 2)
        {
            if (prev_c == width - 1)
            {
                min_seam_track[r] = width - 1;
            }
            else
            {
                min_seam_track[r] = prev_c + 1;
            }
        }
    }
    CHECK(cudaMemcpy(d_min_seam_track, min_seam_track, byteSeam, cudaMemcpyHostToDevice));
    // Call kernel
    removeMinSeamOnDevice<<<gridSize, blockSize>>>(width, height, d_in, d_min_seam_track, d_out);
    CHECK(cudaDeviceSynchronize());
    // Copy data
    CHECK(cudaMemcpy(outPixels, d_out, byteOut, cudaMemcpyDeviceToHost));

    writePnm(outPixels, 3, width - 1, height, fileName);

    for (int i = 0; i < nStreams; i++){
      CHECK(cudaStreamDestroy(streams[i]));
    }
    free(streams);
    CHECK(cudaFree(d_filter1));
    CHECK(cudaFree(d_filter2));

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_grayscale));
    CHECK(cudaFree(d_convo));
    CHECK(cudaFree(d_minE));
    CHECK(cudaFree(d_backtrack));
    CHECK(cudaFree(d_min_seam_track));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_vMinE));

}

// ======================================== MAIN ========================================
int main(int argc, char **argv)
{
    // Args
    dim3 blockSize(32, 32);
    char *fileName = argv[1];

    int mode = atoi(argv[2]);
    int numSeamsToCarve = 1;

    if (mode != 0)
    {
        numSeamsToCarve = atoi(argv[3]);
    }
    if (mode != 0 && mode != 1)
    {
        blockSize.x = atoi(argv[4]);
        blockSize.y = atoi(argv[5]);
    }

    // Filters
    float noFilter[] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0};
    float identityFilter[] = {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0};
    float gaussianBlur[] = {
        1.0 / 16, 2.0 / 16, 1.0 / 16,
        2.0 / 16, 4.0 / 16, 2.0 / 16,
        1.0 / 16, 2.0 / 16, 1.0 / 16};
    float laplacian[] = {
        0, -1, 0,
        -1, 4, -1,
        0, -1, 0};
    float sobel_x[] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1};
    float sobel_y[] = {
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1};
    float temp = noFilter[0]; // temp
    temp = identityFilter[0]; // temp
    temp = gaussianBlur[0];   // temp
    temp = laplacian[0];      // temp
    temp = sobel_x[0];        // temp
    temp = sobel_y[0];        // temp
    temp = temp;
    int filterWidth = 3;
    float *filter1 = (float *)malloc(filterWidth * filterWidth * sizeof(float));
    float *filter2 = (float *)malloc(filterWidth * filterWidth * sizeof(float));
    for (int i = 0; i < filterWidth; i++)
    {
        for (int j = 0; j < filterWidth; j++)
        {
            filter1[i * filterWidth + j] = sobel_x[i * filterWidth + j];
            filter2[i * filterWidth + j] = sobel_y[i * filterWidth + j];
        }
    }
    CHECK(cudaMemcpyToSymbol(dc_filter1, filter1, filterWidth * filterWidth * sizeof(float))); // CMEM
    CHECK(cudaMemcpyToSymbol(dc_filter2, filter2, filterWidth * filterWidth * sizeof(float))); // CMEM

    int numChannels, width, height;

    // Host Memories
    uint8_t *inPixels;
    readPnm(fileName, numChannels, width, height, inPixels); // Just to get raw width and height
    uint8_t *grayscalePixels = (uint8_t *)malloc(width * height * 1 * sizeof(uint8_t));
    uint8_t *convoPixels = (uint8_t *)malloc(width * height * 1 * sizeof(uint8_t));
    uint8_t *minEnergies = (uint8_t *)malloc(width * height * 1 * sizeof(uint8_t));
    int *minEnergiesInt = (int *)malloc(width * height * 1 * sizeof(int));
    uint8_t *backtrack = (uint8_t *)malloc(width * height * 1 * sizeof(uint8_t));
    int *vMinEnergies = (int *)malloc(width * sizeof(int));
    int *min_seam_track = (int *)malloc(height * sizeof(int));
    uint8_t *outPixels = (uint8_t *)malloc((width - 1) * height * numChannels * sizeof(uint8_t)); // width-1: means that the output image will be cut 1 pixel (the min seam)

    // Device Memories
    uint8_t *d_inPixels;
    uint8_t *d_grayscalePixels;
    uint8_t *d_convoPixels;
    float *d_filter1, *d_filter2;
    uint8_t *d_minEnergies, *d_backtrack;
    int *d_vMinEnergies;
    int *d_min_seam_track;
    uint8_t *d_outPixels;

    size_t nBytes_inPixels = width * height * numChannels * sizeof(uint8_t);
    size_t nBytes_grayscalePixels = width * height * 1 * sizeof(uint8_t);
    size_t nBytes_convoPixels = width * height * 1 * sizeof(uint8_t);
    size_t nBytes_filter = filterWidth * filterWidth * sizeof(float);
    size_t nBytes_minEnergies = width * height * 1 * sizeof(uint8_t);
    size_t nBytes_backtrack = width * height * 1 * sizeof(uint8_t);
    size_t nBytes_vMinEnergies = width * sizeof(int);
    size_t nBytes_min_seam_track = height * sizeof(int);
    size_t nBytes_outPixels = (width - 1) * height * numChannels * sizeof(uint8_t);

    CHECK(cudaMalloc(&d_inPixels, nBytes_inPixels));
    CHECK(cudaMalloc(&d_grayscalePixels, nBytes_grayscalePixels));
    CHECK(cudaMalloc(&d_convoPixels, nBytes_convoPixels));
    CHECK(cudaMalloc(&d_filter1, nBytes_filter));
    CHECK(cudaMalloc(&d_filter2, nBytes_filter));
    CHECK(cudaMalloc(&d_minEnergies, nBytes_minEnergies));
    CHECK(cudaMalloc(&d_backtrack, nBytes_backtrack));
    CHECK(cudaMalloc(&d_vMinEnergies, nBytes_vMinEnergies));
    CHECK(cudaMalloc(&d_min_seam_track, nBytes_min_seam_track));
    CHECK(cudaMalloc(&d_outPixels, nBytes_outPixels));

    if (mode != 0)
    {
        // =========================================================================
        GpuTimer timer;
        timer.Start();
        // --------------------- SEAM CARVING - Remove N seams ---------------------
        printf("Seam Carving is running...\n");

        int last_i = 0;
        for (int i = 0; i < numSeamsToCarve; i++)
        {
            // Read input RGB image file
            readPnm(fileName, numChannels, width, height, inPixels);
            nBytes_inPixels = width * height * numChannels * sizeof(uint8_t);
            // SEAM CARVING - Remove 1 seam
            if (mode == 1) // Host
            {
                seamCarvingOnHost(inPixels, fileName, numChannels, width, height, filter1, filter2, filterWidth, blockSize, grayscalePixels, convoPixels, minEnergies, backtrack, vMinEnergies, min_seam_track, outPixels, minEnergiesInt);
            }
            else if (mode == 2) // Device
            {
                seamCarvingOnDevice(inPixels, fileName, numChannels, width, height, filter1, filter2, filterWidth, blockSize, grayscalePixels, convoPixels, minEnergies, backtrack, vMinEnergies, min_seam_track, outPixels, d_inPixels, nBytes_inPixels, d_grayscalePixels, nBytes_grayscalePixels, d_convoPixels, nBytes_convoPixels, d_filter1, d_filter2, nBytes_filter, d_backtrack, nBytes_backtrack, d_minEnergies, nBytes_minEnergies, d_vMinEnergies, nBytes_vMinEnergies, d_min_seam_track, nBytes_min_seam_track, d_outPixels, nBytes_outPixels, minEnergiesInt);
            }
            else if (mode == 3) // Device (Optimize 1)
            {
                seamCarvingOnDeviceOpt1(inPixels, fileName, numChannels, width, height, filterWidth, blockSize, grayscalePixels, convoPixels, minEnergies, backtrack, vMinEnergies, min_seam_track, outPixels, d_inPixels, nBytes_inPixels, d_grayscalePixels, nBytes_grayscalePixels, d_convoPixels, nBytes_convoPixels, d_backtrack, nBytes_backtrack, d_minEnergies, nBytes_minEnergies, d_vMinEnergies, nBytes_vMinEnergies, d_min_seam_track, nBytes_min_seam_track, d_outPixels, nBytes_outPixels, minEnergiesInt);
            }
            else if (mode == 4) // Device (Optimize 2)
            {
                // printf("> Not implemented yet 🙀");
                seamCarvingOnDeviceOpt2(inPixels, fileName, numChannels, width, height, atoi(argv[6]), filter1, filter2, filterWidth, blockSize, grayscalePixels, convoPixels, minEnergies, backtrack, vMinEnergies, min_seam_track, outPixels, minEnergiesInt);
            }
            // printf("> i = %d (%d x %d) finished\n", i, width, height);
            last_i = i;
        }
        printf("> Removed %d seams\n", last_i + 1);

        // -------------------------------------------------------------------------
        timer.Stop();
        float time = timer.Elapsed();
        if (mode == 1) // Host
        {
            printf("> Using HOST\n");
        }
        else if (mode == 2) // Device
        {
            printf("> Using DEVICE\n");
        }
        else if (mode == 3) // Device (Optimize 1)
        {
            printf("> Using DEVICE (OPTIMIZE 1)\n");
        }
        else if (mode == 4) // Device (Optimize 2)
        {
            printf("> Using DEVICE (OPTIMIZE 2)\n");
        }
        printf("> Processing time: %f ms\n", time);
        // =========================================================================
        printf(">>> Completed!\n");
    }
    else
    {
        readPnm(fileName, numChannels, width, height, inPixels);
        nBytes_inPixels = width * height * numChannels * sizeof(uint8_t);
        seamCarvingOnHost(inPixels, fileName, numChannels, width, height, filter1, filter2, filterWidth, blockSize, grayscalePixels, convoPixels, minEnergies, backtrack, vMinEnergies, min_seam_track, outPixels, minEnergiesInt);
    }

    // ----- Free Host Memories
    free(inPixels);
    free(filter1);
    free(filter2);
    free(grayscalePixels);
    free(convoPixels);
    free(minEnergies);
    free(minEnergiesInt);
    free(backtrack);
    free(vMinEnergies);
    free(min_seam_track);
    free(outPixels);
    // ----- Free Device Memories
    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_grayscalePixels));
    CHECK(cudaFree(d_convoPixels));
    CHECK(cudaFree(d_filter1));
    CHECK(cudaFree(d_filter2));
    CHECK(cudaFree(d_minEnergies));
    CHECK(cudaFree(d_backtrack));
    CHECK(cudaFree(d_vMinEnergies));
    CHECK(cudaFree(d_min_seam_track));
    CHECK(cudaFree(d_outPixels));
}
