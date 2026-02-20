/**
 * Shared Memory Producer
 *
 * Writes data to POSIX shared memory with semaphore synchronization.
 * Run this before shm_consumer.
 *
 * Build: make
 * Usage: ./shm_producer
 *
 * Note: Link with -lrt -lpthread on Linux
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>
#include <time.h>

#define SHM_NAME  "/study_shm"
#define SEM_READY "/study_sem_ready"
#define SEM_DONE  "/study_sem_done"

#define NUM_ITEMS 10

typedef struct {
    int    id;
    double value;
    char   label[64];
} item_t;

typedef struct {
    int    count;
    item_t items[NUM_ITEMS];
} shared_data_t;

int main(void) {
    srand(time(NULL));

    /* Create shared memory */
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0) {
        perror("shm_open");
        exit(EXIT_FAILURE);
    }
    ftruncate(shm_fd, sizeof(shared_data_t));

    shared_data_t *shm = mmap(NULL, sizeof(shared_data_t),
                               PROT_READ | PROT_WRITE,
                               MAP_SHARED, shm_fd, 0);
    if (shm == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }

    /* Create semaphores */
    sem_t *sem_ready = sem_open(SEM_READY, O_CREAT, 0666, 0);
    sem_t *sem_done  = sem_open(SEM_DONE,  O_CREAT, 0666, 0);

    /* Produce data */
    shm->count = NUM_ITEMS;
    for (int i = 0; i < NUM_ITEMS; i++) {
        shm->items[i].id = i + 1;
        shm->items[i].value = (double)(rand() % 10000) / 100.0;
        snprintf(shm->items[i].label, sizeof(shm->items[i].label),
                 "item_%03d", i + 1);
    }

    printf("Producer: wrote %d items to shared memory\n", NUM_ITEMS);
    for (int i = 0; i < NUM_ITEMS; i++) {
        printf("  [%d] %s = %.2f\n",
               shm->items[i].id, shm->items[i].label,
               shm->items[i].value);
    }

    /* Signal consumer that data is ready */
    sem_post(sem_ready);
    printf("Producer: signaled consumer, waiting for acknowledgment...\n");

    /* Wait for consumer to finish */
    sem_wait(sem_done);
    printf("Producer: consumer acknowledged. Cleaning up.\n");

    /* Cleanup */
    sem_close(sem_ready);
    sem_close(sem_done);
    munmap(shm, sizeof(shared_data_t));
    close(shm_fd);

    return 0;
}
