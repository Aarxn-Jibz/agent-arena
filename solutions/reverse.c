/* 003: reverse a string (correct) */
#include <stdio.h>
#include <string.h>

int main(void) {
    char s[256];
    int i;
    scanf("%s", s);
    for (i = (int)strlen(s) - 1; i >= 0; i--) putchar(s[i]);
    putchar('\n');
    return 0;
}