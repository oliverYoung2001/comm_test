#include <pthread.h>
#include <iostream>
#include<unistd.h>               // for linux 
#include <cstdio>

using namespace std;
 
#define NUM_THREADS 5
 
// 线程的运行函数
void* say_hello(void* args)
{
    cout << "Hello Runoob!" << endl;
    fflush(stdout);
    sleep(1);   // s
    return 0;
}
 
int main()
{
    // 定义线程的 id 变量，多个变量使用数组
    pthread_t tids[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; ++i)
    {
        //参数依次是：创建的线程id，线程参数，调用的函数，传入的函数参数
        int ret = pthread_create(&tids[i], NULL, say_hello, NULL);
        printf("Thread: %d\n", i);
        if (ret != 0)
        {
           cout << "pthread_create error: error_code=" << ret << endl;
        }
    }
    //等各个线程退出后，进程才结束，否则进程强制结束了，线程可能还没反应过来；
    for (int i = 0; i < NUM_THREADS; ++ i) {
        pthread_join(tids[i], nullptr);
    }
    printf("%d threads are all done !!!\n", NUM_THREADS);
    // pthread_exit(NULL);

}