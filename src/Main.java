import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main{
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int n = Integer.parseInt(br.readLine());
        LRUCache lru = new LRUCache(n);
        String str = null;
        if (n == 0){
            while ((str = br.readLine())!= null){
                if (str.charAt(0) == 'g')
                    System.out.println(-1);
            }
            return;
        }
        while((str = br.readLine()) != null){
            String[] strs = str.split(" ");
            if(strs[0].equals("p")){
                int key = Integer.parseInt(strs[1]);
                int value = Integer.parseInt(strs[2]);
                lru.put(key, value);
            }else if(strs[0].equals("g")){
                int key = Integer.parseInt(strs[1]);
                System.out.println(lru.get(key));
            }
        }
    }


}

class LRUCache{
    Map<Integer, DoubleList> cache = new HashMap<>();
    int size;
    int capacity;
    DoubleList head;
    DoubleList tail;
    public LRUCache(int capacity){
        this.size = 0;
        this.capacity = capacity;
        head = new DoubleList();//伪头部
        tail = new DoubleList();//伪尾部
        head.next = tail;
        tail.pre = head;
    }
    public int get(int key){
        DoubleList node = cache.get(key);
        if(node == null){
            return -1;
        }
        moveToHead(node);
        return node.value;
    }
    public void put(int key, int value){
        DoubleList node = cache.get(key);
        if(node == null){
            DoubleList newNode = new DoubleList(key, value);
            cache.put(key, newNode);
            addToHead(newNode);
            size++;
            if(size > capacity){
                DoubleList tailNode = removeTail();
                cache.remove(tailNode.key);//删除尾部
                size--;
            }
        }else{
            node.value = value;//替换原有值
//            moveToHead(node);//移动到头部
        }
    }

    private DoubleList removeTail(){
        DoubleList realTail = tail.pre;
        removeNode(realTail);
        return realTail;
    }

    private void moveToHead(DoubleList node){
        removeNode(node);
        addToHead(node);
    }

    private void removeNode(DoubleList node){
        node.pre.next = node.next;
        node.next.pre = node.pre;
    }

    private void addToHead(DoubleList node){
        node.pre = head;
        node.next = head.next;
        head.next.pre = node;
        head.next = node;
    }

    public class DoubleList{
        int key;
        int value;
        DoubleList pre;
        DoubleList next;
        public DoubleList(){}
        public DoubleList(int key, int value){
            this.key = key;
            this.value = value;
        }
    }
}
