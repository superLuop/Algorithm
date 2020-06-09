import java.util.*;
public class JosephRing {
    public static void main(String[] args){
        Scanner in = new Scanner(System.in);
        while (in.hasNext()){
            int n = in.nextInt();
            int m = in.nextInt();
            calculate(n, m);
        }
    }

    private static void calculate(int n, int m) {
        List<Integer> res = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            res.add(i);
        }
        int index = 0;
        while (res.size() > 1){
            index  = (index + m - 1) % res.size();
            System.out.print(res.get(index) + " ");
            res.remove(index);
        }
        System.out.print(res.get(0));
    }
    private static void calculate2(int n, int m) {
        Node pre = new Node(1);
        Node cur = pre;
        for (int i = 2; i <= n; i++) {
            cur.next = new Node(i);
            cur = cur.next;
        }
        cur.next = pre;
        int count = 0;
        while (cur != cur.next){
            if (++count == m){
                System.out.print(cur.next.val + " ");
                cur.next = cur.next.next;
                count = 0;
            }else {
                cur = cur.next;
            }
        }
        System.out.print(cur.val);
    }

    public static class Node{
        int val;
        Node next;
        Node(int val){
            this.val = val;
        }
    }
}