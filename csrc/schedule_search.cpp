#include <iostream>
#include <cstdio>
#include <cstring>
#define INDEX2(x, y) ((x) * comm_size + (y))

bool* P;
int comm_size, schedule_count;
int* r_m;

void dfs(int r, int x) {
    if (r > comm_size) {
        ++ schedule_count;
        // printf("Schedule %d:\n", schedule_count);
        // for (int i = 0; i < comm_size; ++ i) {
        //     for (int j = 0; j < comm_size; ++ j) {
        //         printf("%d ", r_m[INDEX2(i, j)]);
        //     }
        //     puts("");
        // }
        return;
    }
    if (x == 0) {
        P[INDEX2(r - 1, r - 1)] = 1;
        r_m[INDEX2(x, r - 1)] = r;
        dfs(r + (x + 1 == comm_size), (x + 1) % comm_size);
        r_m[INDEX2(x, r - 1)] = 0;
        P[INDEX2(r - 1, r - 1)] = 0;
        return;
    }
    // printf("%d %d\n", r, x);
    for (int y = 0; y < comm_size; ++ y) {
        // printf("%d %d\n", y, r_m[INDEX2(x, y)]);
        if (r_m[INDEX2(x, y)] == 0) {
            if (! P[INDEX2(r - 1, y)]) {
                P[INDEX2(r - 1, y)] = 1;
                r_m[INDEX2(x, y)] = r;
                dfs(r + (x + 1 == comm_size), (x + 1) % comm_size);
                r_m[INDEX2(x, y)] = 0;
                P[INDEX2(r - 1, y)] = 0;
            }
        }
    }
}

int main() {
    for (comm_size = 4; comm_size <= 8; ++ comm_size) {
        schedule_count = 0;
        r_m = new int[comm_size * comm_size];
        P = new bool[comm_size * comm_size];
        memset(r_m, 0, comm_size * comm_size * sizeof(int));
        memset(P, 0, comm_size * comm_size * sizeof(bool));
        for (int i = 0; i < comm_size; ++ i) {
            r_m[INDEX2(i, i)] = 1;
        }
        printf("comm_size: %d\n", comm_size);
        dfs(2, 0);
        printf("schedule_count = %d\n", schedule_count);
        puts("");
        delete[] P;
        delete[] r_m;
    }
    return 0;
}