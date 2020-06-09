import java.util.Scanner;

public class Flowers {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int len = in.nextInt();
        int[] arr = new int[len];
        for (int i = 0; i < len; i++) {
            arr[i] = in.nextInt();
        }
        System.out.println(PlaceFlowers(arr));
    }
    private static int PlaceFlowers(int[] flowers){
        int i = 0, count = 0;
        while (i < flowers.length){
            if (flowers[i] == 0 && (i == 0 || flowers[i - 1] == 0) && (i == flowers.length - 1 || flowers[i + 1] == 0)){
                flowers[i] = 1;
                count++;
            }
            i++;
        }
        return count;
    }
}
