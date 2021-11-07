class Heap:
    def __init__(self, cmp=lambda a,b: a>b):
        self.heap = [None]
        self.cmp = cmp
    
    def swap(self, a, b):
        self.heap[a], self.heap[b] = self.heap[b], self.heap[a]
    
    def push(self, x):
        self.heap.append(x)
        self.pushup(len(self.heap)-1)
    
    def pop(self):
        if len(self.heap) <= 1: return
        self.swap(1, -1)
        self.heap.pop()
        self.pushdown(1)
    
    def pushup(self, idx):
        parent = idx//2
        while parent >= 1:
            if self.cmp(self.heap[idx],self.heap[parent]):
                self.swap(idx, parent)
                idx = parent
                parent = idx // 2
            else:
                break
    
    def pushdown(self, idx):
        size = len(self.heap)
        heap = self.heap
        while idx < size:
            max_idx = idx
            if idx * 2 < size and self.cmp(heap[idx*2],heap[max_idx]):
                max_idx = 2*idx
            if idx * 2 + 1 < size and self.cmp(heap[idx*2+1],heap[max_idx]):
                max_idx = 2*idx+1
            if max_idx != idx:
                self.swap(max_idx, idx)
                idx = max_idx
            else:
                break

if __name__ == '__main__':
    heap = Heap(lambda a,b: a<b)
    n = int(input())
    for i in range(n):
        line = input().strip()
        if line.startswith("1"):
            _, x = line.split(' ')
            x = int(x)
            heap.push(x)
        elif line == "2":
            print(heap.heap[1])
        else:
            heap.pop()

