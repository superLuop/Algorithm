import java.util.*;

public class ExchangeStation {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNext()){
            int n = in.nextInt();
            int[][] arr = new int[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    arr[i][j] = in.nextInt();
                }
            }
            List<int[]> list0 = new ArrayList<>();
            List<int[]> list1 = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (arr[i][j] == 0){
                        list0.add(new int[]{i, j});
                    }else {
                        list1.add(new int[]{i, j});
                    }
                }
            }
            int minDis = Integer.MAX_VALUE;
            if (list0.size() == 0){
                System.out.println(-1);
                return;
            }
            for (int[] tmp : list0) {
                int dis = 0;
                int x = tmp[0], y = tmp[1];
                for (int[] tmp1 : list1) {
                    int nx = tmp1[0], ny = tmp1[1];
                    dis += Math.abs(nx - x) + Math.abs(ny - y);
                }
                if (minDis > dis)
                    minDis = dis;
            }
            System.out.println(minDis);
        }
    }
}
