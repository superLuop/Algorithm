
import java.util.Scanner;

/**
4
2 4 8
3 4 6
5 8 9 11
1 6 7 10 12
 */
public class sortedNList {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int n = Integer.parseInt(in.nextLine());
        String[] strs = new String[n];
        ListNode[] lists = new ListNode[n];
        ListNode pre = new ListNode(-1);
        ListNode cur = pre;
        for (int i = 0; i < n; i++) {
            strs[i] = in.nextLine();
            String[] split = strs[i].split(" ");
            for (String s : split) {
                int data = Integer.parseInt(s);
                cur.next = new ListNode(data);
                cur = cur.next;
                lists[i] = pre.next;
            }
            cur = pre;
        }
        cur = mergeList(lists);
        while (cur != null){
            System.out.print(cur.val + " ");
            cur = cur.next;
        }
    }

    private static ListNode mergeList(ListNode[] lists) {
        if (lists == null || lists.length == 0){
            return null;
        }
        return binaryConquer(lists, 0, lists.length - 1);
    }

    private static ListNode binaryConquer(ListNode[] lists, int start, int end) {
        if (start < end){
            int mid = start + (end - start) / 2;
            ListNode l1 = binaryConquer(lists, start, mid);
            ListNode l2 = binaryConquer(lists, mid + 1, end);
            return mergeTwoList(l1, l2);
        }
        return lists[start];
    }

    private static ListNode mergeTwoList(ListNode l1, ListNode l2) {
        ListNode pre = new ListNode(-1);
        ListNode cur = pre;
        while (l1 != null && l2 != null){
            if (l1.val < l2.val){
                cur.next = l1;
                l1 = l1.next;
            }else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        cur.next = l1 == null ? l2 : l1;
        return pre.next;
    }

    public static class ListNode{
        int val;
        ListNode next;
        public ListNode(int val){
            this.val = val;
        }
    }
}
