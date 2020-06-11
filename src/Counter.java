import java.util.*;

public class Counter{
    public static void main(String[] args){
        Scanner in = new Scanner(System.in);
        while(in.hasNext()){
            long time = in.nextLong();
            long value = 3;
            while(time - value > 0){
                time -= value;
                value <<= 1;
            }
            value -= time - 1;
            System.out.println(value);
        }
    }
}
