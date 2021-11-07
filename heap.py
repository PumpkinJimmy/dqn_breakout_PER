class Heap:
    def __init__(self, cmp=lambda a,b: a>b):
        self.data = [None]
        self.cmp = cmp
    
    def swap(self, a, b):
        self.data[a], self.data[b] = self.data[b], self.data[a]
    
    def push(self, x):
        self.data.append(x)
        self.pushup(len(self.data)-1)
    
    def pop(self):
        if len(self.data) <= 1:
            return False
        self.swap(1, -1)
        self.data.pop()
        self.pushdown(1)
        return True
    
    def replace(self, idx, x):
        if idx < 1 or idx >= len(self.data):
            return False
        self.data[idx] = x
        self.pushup(idx)
        self.pushdown(idx)
    
    def pushup(self, idx):
        parent = idx//2
        while parent >= 1:
            if self.cmp(self.data[idx],self.data[parent]):
                self.swap(idx, parent)
                idx = parent
                parent = idx // 2
            else:
                break
    
    def pushdown(self, idx):
        size = len(self.data)
        heap = self.data
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
            print(heap.data[1])
        else:
            heap.pop()

