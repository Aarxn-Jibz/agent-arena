#include <stdio.h>
#include <stdlib.h>

#define SIZE 256
#define SEED 1051

int main() {
    char *buffer;
    int i;
    int size = SIZE;
    int seed = SEED;

    // Allocate buffer
    buffer = malloc(size * sizeof(char));
    if (buffer == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Read input from standard input
    fread(buffer, 1, size, stdin);
    fclose(stdin);

    // Decompress input
    for (i = 0; i < size; i++) {
        buffer[i] = buffer[i] ^ seed;
    }
    fwrite(buffer, 1, size, stdout);

    // Free allocated memory
    free(buffer);

    return 0;
}
