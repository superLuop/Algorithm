import java.util.Scanner;

public class MaxSubArray {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNext()){
            int N = in.nextInt();
            int[] arr = new int[N];
            for (int i = 0; i < N; i++) {
                arr[i] = in.nextInt();
            }
            int maxSum = findMaxSum(arr);
            System.out.println(maxSum);
        }
    }

    private static int findMaxSum(int[] arr) {
        if (arr.length == 0){
            return 0;
        }
        int[] dp = new int[arr.length];
        dp[0] = arr[0];
        int maxSum = arr[0];
        for (int i = 1; i < arr.length; i++) {
            dp[i] = Math.max(dp[i - 1] + arr[i], arr[i]);
            maxSum = Math.max(dp[i], maxSum);
        }
        return maxSum;
    }
}
