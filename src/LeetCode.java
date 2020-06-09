
import java.util.*;
import java.util.stream.IntStream;

public class LeetCode {
    public static void main(String[] args) {
        LeetCode lc = new LeetCode();
//        System.out.println(lc.longestPalindrome("cbbd"));
//        System.out.println(lc.minWindow("bbaa", "aba"));

//        System.out.println(lc.findMedianSortedArrays(new int[]{4}, new int[]{3, 4}));

//        System.out.println(lc.subarraysDivByK(new int[]{4, 5, 0, -2, -3, 1}, 5));

//        System.out.println(lc.rob(new int[]{2,7,9,3,1}));

//        System.out.println(lc.combinationSum2(new int[]{10,1,2,7,6,1,5}, 8));

//        System.out.println(lc.largestRectangleArea(new int[]{0,9}));

        int[][] r1 = new int[][]{{5, 1, 9, 11},{2, 4, 8, 10},{13, 3, 6, 7},{15, 14, 12, 16}};
        int[][] r2 = new int[][]{{1, 2, 3},{4, 5, 6},{7, 8, 9}};
//        lc.rotate(r2);
//        System.out.println(Arrays.deepToString(r2));
//
//        System.out.println(lc.addString("1233275897430683046", "79842367803486594326"));
//
//        System.out.println(lc.product("123456789", "123456789"));

//        System.out.println(Arrays.deepToString(lc.ArrayProduct(r2, r2)));

//        System.out.println(Arrays.deepToString(lc.ArrayOfN(2)));

//        double r = 9.9955926;
//        System.out.println(String.format("%.2f", r));//保留两位有效数字
//
//        System.out.println(lc.groupAnagrams(new String[]{"eat", "tea", "tan", "ate", "nat", "bat"}));

//        System.out.println(lc.kidsWithCandies(new int[]{12,1,12}, 10));

//        System.out.println(lc.isMatch("adceb", "*a*b"));

//        System.out.println(lc.sumNums(3));

//        System.out.println(Arrays.deepToString(lc.generateMatrix(1)));

//        System.out.println(lc.new21Game(21, 17, 10));

//        System.out.println(lc.removeBoxes(new int[]{1, 3, 2, 2, 2, 3, 4, 3, 1}));

//        System.out.println(Arrays.toString(lc.productExceptSelf(new int[]{1, 2, 3, 4})));

//        System.out.println(lc.robot("RRU", new int[][]{{5,5},{9,4},{9,7},{6,4},{7,0},{9,5},{10,7},{1,1},{7,5}}, 1486, 743));

//        System.out.println(lc.getMaxSubString("asddghjukihjughjkssfghyj", "adhasjkssdghjuoop"));
//
//        System.out.println(lc.getMaxLenCommonString("asddghjukihjughjkssfghyj", "adhasjkssdghjuoop"));

//        System.out.println(lc.longestConsecutive(new int[]{100, 4, 200, 1, 3, 2}));

//        List<String> wordList = Arrays.asList("hot", "dot", "dog", "lot", "log", "cog");
//        System.out.println(lc.findLadders("hit", "cog", wordList));

//        System.out.println(Arrays.toString(lc.gardenNoAdj(3, new int[][]{{1,2}, {2,3},{3,1}})));

//        System.out.println(lc.equationPossible(new String[]{"a==b", "b!=c", "c==a"}));

//        System.out.println(lc.longestValidParentheses("(()"));
//
//        lc.nextPermutation(new int[]{1,3,5,7,2,4,6});

//        System.out.println(lc.permute(new int[]{1, 2, 3}));

//        System.out.println(lc.translateNum(12258));

        System.out.println(lc.exist(new char[][]{{'A', 'B','C','E'}, {'S', 'F','C','S'},{'A','D','E','E'}}, "ABCCED"));
    }

    /**
     * 矩阵中的路径 -- 回溯法
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        int m = board.length;
        int n = board[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (isExist(board, m, n, i, j, word, 0)){
                    return true;
                }
            }
        }
        return false;
    }

    private boolean isExist(char[][] board, int m, int n, int i, int j, String word, int index) {
        if (i < 0 || i >=m || j < 0 || j >= n || board[i][j] != word.charAt(index)){
            return false;
        }
        if (index == word.length() - 1){
            return true;
        }
        char tmp = board[i][j];
        board[i][j] = '&';
        boolean res = isExist(board, m, n, i - 1, j, word, index + 1) ||
                isExist(board, m, n, i + 1, j, word, index + 1) ||
                isExist(board, m, n, i, j - 1, word, index + 1) ||
                isExist(board, m, n, i, j + 1, word, index + 1);
        board[i][j] = tmp;
        return res;
    }

    /**
     * 把数字翻译成字符串 -- 动态规划
     * @param num
     * @return
     */
    public int translateNum(int num) {
        String s = String.valueOf(num);
        int n = s.length();
        int[] dp = new int[n];
        dp[0] = 1;
        for (int i = 1; i < n; i++) {
            int tmp = Integer.parseInt(s.substring(i - 1, i + 1));
            if (tmp >= 10 && tmp <= 25){
                dp[i] = dp[i - 1] + (i >= 2 ? dp[i - 2] : 1);
            }else {
                dp[i] = dp[i - 1];
            }
        }
        return dp[n - 1];
    }

    /**
     * 全排列
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        getPermute(nums, 0, nums.length, res);
        return res;
    }

    private void getPermute(int[] nums, int start, int end, List<List<Integer>> res) {
        if (start == end){
            List<Integer> list = new ArrayList<>();
            for (int num : nums) {
                list.add(num);
            }
            res.add(list);
        }
        for (int i = start; i < end; i++) {
            swap(nums, i, start);
            getPermute(nums, start + 1, end, res);
            swap(nums, i, start);
        }
    }

    /**
     * 下一个排列
     * @param nums
     */
    public void nextPermutation(int[] nums) {
        int n = nums.length;
        int i = n - 2;
        while(i >= 0 && nums[i + 1] <= nums[i]){
            i--;
        }
        if(i >= 0){
            int j = n - 1;
            while(j >= 0 && nums[j] <= nums[i]){
                j--;
            }
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
        System.out.println(Arrays.toString(nums));
    }
    private void reverse(int[] nums, int i){
        int j = nums.length - 1;
        while(i < j){
            int tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
            i++;
            j--;
        }
    }
    private void swap(int[] nums, int i, int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    /**
     * 最长有效括号 -- 栈
     * @param s
     * @return
     */
    public int longestValidParentheses(String s) {
        if (s == null || s.length() <= 1){
            return 0;
        }
        int n = s.length();
        int len = 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) == '('){
                stack.push(i);
            }else {
                stack.pop();
                if (stack.isEmpty()){
                    stack.push(i);
                }else {
                   len = Math.max(len, i - stack.peek());
                }
            }
        }
        return len;
    }
    //动态规划
    public int longestValidParentheses2(String s) {
        if (s == null || s.length() <= 1){
            return 0;
        }
        int n = s.length();
        int[] dp = new int[n];
        int len = 0;
        for (int i = 1; i < n; i++) {
            if (s.charAt(i) == ')'){
                if (s.charAt(i - 1) == '('){
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                }else if (i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '('){
                    dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
            }
            len = Math.max(dp[i], len);
        }
        return len;
    }
    //超时
    public int longestValidParentheses3(String s) {
        if (s == null || s.length() <= 1){
            return 0;
        }
        int n = s.length();
        int len = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 2; j <= n; j += 2) {
                String s1 = s.substring(i, j);
                if (isValid(s1)){
                    len = Math.max(len, j - i);
                }
            }
        }
        return len;
    }

    private boolean isValid(String s1) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s1.length(); i++) {
            if (s1.charAt(i) == '('){
                stack.push('(');
            }else if (!stack.isEmpty() && stack.peek() == '('){
                stack.pop();
            }else {
                return false;
            }
        }
        return stack.isEmpty();
    }

    /**
     * 等式方程的可满足性 -- 并查集
     * @param equations
     * @return
     */
    public boolean equationPossible(String[] equations){
        if(equations == null || equations.length == 0){
            return true;
        }
        int[] parent = new int[26];
        for (int i = 0; i < 26; i++) {
            parent[i] = i;
        }
        for (String str : equations) {
            if (str.charAt(1) == '='){
                int index1 = str.charAt(0) - 'a';
                int index2 = str.charAt(3) - 'a';
                union(parent, index1, index2);
            }
        }
        for (String str : equations) {
            if (str.charAt(1) == '!'){
                int index1 = str.charAt(0) - 'a';
                int index2 = str.charAt(3) - 'a';
                if (find(parent, index1) == find(parent, index2)){
                    return false;
                }
            }
        }
        return true;
    }

    private void union(int[] parent, int index1, int index2) {
        parent[find(parent, index1)] = find(parent, index2);
    }

    private int find(int[] parent, int index) {
        while (parent[index] != index){
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        return index;
    }

    /**
     * 不临接植花
     * @param N
     * @param paths
     * @return
     */
    public int[] gardenNoAdj(int N, int[][] paths) {
        int[] res = new int[N];
        Map<Integer, Set<Integer>> map = new HashMap<>();
        for (int i = 0; i < N; i++) {
            map.put(i, new HashSet<>());
        }
        for (int[] path : paths) {
            int p1 = path[0] - 1;
            int p2 = path[1] - 1;
            map.get(p1).add(p2);
            map.get(p2).add(p1);
        }
        for (int i = 0; i < N; i++) {
            boolean[] used = new boolean[5];
            for (int adj : map.get(i)) {
               used[res[adj]] = true;
            }
            for (int j = 1; j <= 4; j++) {
                if (!used[j]){
                    res[i] = j;
                }
            }
        }
        return res;
    }

    /**
     * 单词接龙II
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    //注：List.contains()效率低于Set.contains()
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        List<List<String>> res = new ArrayList<>();
        if (!wordList.contains(endWord)){
            return res;
        }
        Set<String> visited = new HashSet<>();
        Set<String> dist = new HashSet<>(wordList);
        Queue<List<String>> q = new LinkedList<>();
        List<String> path = new ArrayList<>();
        path.add(beginWord);
        q.offer(path);
        visited.add(beginWord);
        boolean isFind = false;
        while (!q.isEmpty() && !isFind){
            int size = q.size();
            Set<String> subVisited = new HashSet<>();
            for (int i = 0; i < size; i++) {
                List<String> p = q.poll();
                assert p != null;
                String lastWord = p.get(p.size() - 1);
                char[] chars = lastWord.toCharArray();
                for (int j = 0; j < chars.length; j++) {
                    char tmp = chars[j];
                    for (char c = 'a'; c <= 'z'; c++) {
                        if (c == chars[j]) {
                            continue;
                        }
                        chars[j] = c;
                        String nextWord = new String(chars);
                        if (dist.contains(nextWord) && !visited.contains(nextWord)){
                            List<String> list = new ArrayList<>(p);
                            list.add(nextWord);
                            if (nextWord.equals(endWord)){
                                isFind = true;
                                res.add(list);
                            }
                            q.add(list);
//                            p.remove(p.size() - 1);
                            subVisited.add(nextWord);
                        }
                    }
                    chars[j] = tmp;
                }
            }
            visited.addAll(subVisited);
        }
        return res;
    }

    public List<List<String>> findLadders2(String beginWord, String endWord, List<String> wordList) {
        List<List<String>> res = new ArrayList<>();
        if (!wordList.contains(endWord)){
            return res;
        }
        wordBFS(beginWord, endWord, wordList, res);
        return res;
    }

    private void wordBFS(String beginWord, String endWord, List<String> wordList, List<List<String>> res) {
        Queue<List<String>> q = new LinkedList<>();
        List<String> path = new ArrayList<>();
        path.add(beginWord);
        q.offer(path);
        Set<String> dist = new HashSet<>(wordList);
        Set<String> visited = new HashSet<>();
        visited.add(beginWord);
        boolean isFind = false;
        while (!q.isEmpty()){
            int size = q.size();
            Set<String> subVisited = new HashSet<>();
            for (int i = 0; i < size; i++) {
                List<String> p = q.poll();
                assert p != null;
                String tmp = p.get(p.size() - 1);
                List<String> nextWords = getNextWords(tmp, dist);
                for (String next : nextWords) {
                    if (!visited.contains(next)){
                        p.add(next);
                        if (next.equals(endWord)){
                            res.add(new ArrayList<>(p));
                            isFind = true;
                        }
                        q.offer(new ArrayList<>(p));
                        p.remove(p.size() - 1);
                        subVisited.add(next);
                    }
                }
            }
            visited.addAll(subVisited);
            if (isFind){
                break;
            }
        }

    }

    private List<String> getNextWords(String word, Set<String> dist) {
        List<String> ans = new ArrayList<>();
        char[] chars = word.toCharArray();
        for (char c = 'a'; c <= 'z'; c++) {
            for (int i = 0; i < chars.length; i++) {
                if (c == chars[i]){
                    continue;
                }
                char tmp = chars[i];
                chars[i] = c;
                if (dist.contains(String.valueOf(chars))){
                    ans.add(String.valueOf(chars));
                }
                chars[i] = tmp;
            }
        }
        return ans;
    }

    /**
     * 最长连续序列
     * @param nums
     * @return
     */
    public int longestConsecutive(int[] nums) {
        if(nums == null || nums.length == 0){
            return 0;
        }
        int max = 1;
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (map.containsKey(num)){
                continue;
            }
            int l = map.getOrDefault(num - 1, 0);
            int r = map.getOrDefault(num + 1, 0);
            int tmp = l + r + 1;
            max = Math.max(max, tmp);
            map.put(num, -1);//防止重复

            map.put(num - l, tmp);
            map.put(num + r, tmp);
        }
        return max;
    }

    public int longestConsecutive2(int[] nums) {
        if(nums == null || nums.length == 0){
            return 0;
        }
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int maxLen = 1;
        for (int num : nums) {
            if (!set.contains(num - 1)){
                int cur = num;
                int len = 1;
                while (set.contains(cur + 1)){
                    cur += 1;
                    len++;
                }
                maxLen = Math.max(maxLen, len);
            }
        }
        return maxLen;
    }

    /**
     * 最长公共子序列
     * @param s1
     * @param s2
     * @return
     */
    public int getMaxLenCommonString(String s1, String s2){
        if (s1 == null || s2 == null || s1.length() == 0 || s2.length() == 0){
            return 0;
        }
        int m = s1.length();
        int n = s2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 最长公共子串
     * @param str1
     * @param str2
     * @return
     */
    public String getMaxSubString(String str1, String str2){
        String maxStr ="";
        String str = (str1.length() > str2.length()) ? str1 : str2;//最长串
        String key = str.equals(str1) ? str2 : str1;//最短串

        for(int i = 0; i < key.length(); i++){
            for(int j = key.length(); j > i; j--){
                String temp = key.substring(i, j);
                if(str.contains(temp) && maxStr.length() < temp.length()){
                    maxStr = temp;
                }
            }
        }
        return maxStr;
    }

    /**
     * 机器人大冒险
     * @param command - 指令
     * @param obstacles - 障碍物
     * @param x - 终点x坐标
     * @param y - 终点y坐标
     * @return - 是否完好到达终点（不碰上障碍物）
     */
    public boolean robot(String command, int[][] obstacles, int x, int y){
        //多次循环 找到模式
        //学到了新的存储坐标的方法  左坐标左移30 | 右坐标
        int x0 = 0, y0 = 0;
        Set<Long> ss = new HashSet<>();
        ss.add(((long) x0 << 30) | y0);
        for(int i = 0; i < command.length(); i++){
            char c = command.charAt(i);
            if(c == 'U'){
                y0++;
            }else if (c == 'R'){
                x0++;
            }
            ss.add(((long) x0 << 30) | y0);
        }
        int cir = Math.min( x / x0, y / y0);
        if(!ss.contains(((long) (x - cir * x0) << 30) | (y - cir * y0))){
            return false;
        }
        for(int[] s : obstacles){
            if(s.length != 2) continue;
            int x1 = s[0];
            int y1 = s[1];
            if(x1 > x || y1 > y) continue;
            cir = Math.min(x1 / x0, y1 / y0);
            if(ss.contains(((long) (x1 - cir * x0) << 30) | (y1 - cir * y0))){
                return false;
            }
        }
        return true;
    }

    /**
     * 除自身以外数组的乘积
     * @param nums 数组
     * @return
     */
    public int[] productExceptSelf(int[] nums) {
        if (nums == null || nums.length == 0){
            return new int[]{};
        }
        int n = nums.length;
        int[] out = new int[n];
        int tmp = 1;
        for (int i = 0; i < n; i++) {
            out[i] = tmp;
            tmp *= nums[i];//存储的是当前元素前项乘积
        }
        tmp = 1;
        for (int i = n - 1; i >= 0; i--) {
            out[i] *= tmp;
            tmp *= nums[i];//当前元素后项乘积
        }
        return out;
    }

    public int[] productExceptSelf2(int[] nums) {
        if (nums == null || nums.length == 0){
            return new int[]{};
        }
        int n = nums.length;
        int[] out = new int[n];
        int[] pre = new int[n];
        int[] pos = new int[n];
        pre[0] = 1; pos[n - 1] = 1;
        for (int i = 1, j = n - 2; i < n && j >= 0; i++, j--) {
            pre[i] = pre[i - 1] * nums[i - 1];
            pos[j] = pos[j + 1] * nums[j + 1];
        }
        for (int i = 0; i < n; i++) {
            out[i] = pre[i] * pos[i];
        }
        return out;
    }

    /**
     * 移除盒子
     * @param boxes
     * @return
     */
    public int removeBoxes(int[] boxes) {
        if (boxes == null || boxes.length == 0){
            return 0;
        }
        int n = boxes.length;
        int[][][] dp = new int[n][n][n];
        return  getScore(boxes, dp, 0, n - 1, 0);
    }

    private int getScore(int[] boxes, int[][][] dp, int l, int r, int k) {
        if (l > r){
            return 0;
        }
        if (dp[l][r][k] != 0){
            return dp[l][r][k];
        }
        while (r > l && boxes[r] == boxes[r - 1]){
            r--;
            k++;
        }
        dp[l][r][k] = getScore(boxes, dp, l, r - 1, 0) + (k + 1) * (k + 1);
        for (int i = l; i < r; i++) {
            if (boxes[i] == boxes[r]){
                dp[l][r][k] = Math.max(dp[l][r][k], getScore(boxes, dp,i + 1, r - 1, 0) + getScore(boxes, dp, l, i, k + 1));
            }
        }
        return dp[l][r][k];
    }

    /**
     * 新21点
     * @param N
     * @param K
     * @param W
     * @return
     */
    public double new21Game(int N, int K, int W) {
        double[] dp = new double[N + W + 1];
        for (int i = K; i <= N; i++) {
            dp[i] = 1d;
        }
        double sum = Math.min(N - K + 1, W);
        for (int i = K - 1; i >= 0; i--) {
            dp[i] = sum / W;
            sum += dp[i] - dp[i + W];
        }
        return dp[0];
    }

    /**
     * 螺旋矩阵II
     * @param n
     * @return
     */
    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        int t = 0, b = n - 1, l = 0, r = n - 1;
        int i = 1;
        while (i <= n * n){
            for (int j = l; j <= r; j++) {
                res[t][j] = i++;
            }
            t++;
            for (int j = t; j <= b; j++) {
                res[j][r] = i++;
            }
            r--;
            for (int j = r; j >= l; j--) {
                res[b][j] = i++;
            }
            b--;
            for (int j = b; j >= t; j--) {
                res[j][l] = i++;
            }
            l++;
        }
        return res;
    }

    /**
     * 求 1+2+...+n -- 使用短路与终止
     * 要求:不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
     * @param n
     * @return
     */
    int result = 0;
    public int sumNums(int n) {
        boolean flag = n > 0 && sumNums(n - 1) > 0;
        result += n;
        return result;
    }

    /**
     * 通配符匹配 -- 回溯法
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p){
        if ((s == null && p == null)){
            return true;
        }else if (s == null || p == null){
            return false;
        }
        int m = s.length();
        int n = p.length();
        int i = 0, j = 0, iStar = -1, jStar = -1;
        while (i < m){
            if (j < n && (s.charAt(i) == p.charAt(j) || p.charAt(j) == '?')){
                i++;
                j++;
            }else if (j < n && (p.charAt(j) == '*')){
                iStar = i;//记录位置
                jStar = j++;
            }else if (iStar == -1){
                return false;
            }else {
                i = ++iStar;
                j = jStar + 1;
            }
        }
        for (int k = j; k < n; k++) {
            if (p.charAt(k) != '*'){
                return false;
            }
        }
        return true;
    }

    /**
     * 通配符匹配 -- 动态规划
     * @param s - 待匹配字符串
     * @param p - 模式字符串
     * @return
     */
    public boolean isMatch2(String s, String p){
        if ((s == null && p == null)){
            return true;
        }else if (s == null || p == null){
            return false;
        }
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[n + 1][m + 1];
        dp[0][0] = true;
        for (int i = 1; i <= n; i++) {
            //增、删
            if (p.charAt(i - 1) == '*'){
                int j = 1;
                while ((!dp[i - 1][j - 1]) && (j <= m)){
                    j++;
                }
                dp[i][j - 1] = dp[i - 1][j - 1];
                while (j <= m){
                    dp[i][j++] = true;
                }
            }else if (p.charAt(i - 1) == '?'){ //改
                for (int j = 1; j <= m; j++) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }else {
                for (int j = 1; j <= m; j++) {
                    if (p.charAt(i - 1) == s.charAt(j - 1)){
                        dp[i][j] = dp[i - 1][j - 1];
                    }
                }
            }
        }
        return dp[n][m];
    }
    //时间复杂度O(n^2)
    public boolean isMatch3(String s, String p){
        if ((s == null && p == null)){
            return true;
        }else if (s == null || p == null){
            return false;
        }
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        //空串匹配
        for (int i = 1; i <= n; i++) {
            if (p.charAt(i - 1) == '*'){
                dp[0][i] = dp[0][i - 1];
            }
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*'){
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                }else if (p.charAt(j - 1) == '?' || (s.charAt(i - 1) == p.charAt(j - 1))){
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 拥有最多糖果的孩子
     * @param candies - 糖果数组
     * @param extraCandies - 额外糖果数
     * @return
     */
    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        List<Boolean> res = new ArrayList<>();
        int originalMax = 0;
        for (int num : candies) {
            originalMax = Math.max(originalMax, num);
        }
        for (int candy : candies) {
            res.add(candy + extraCandies >= originalMax);
        }
        return res;
    }

    /**
     * 字母异位词分组
     * @param strs - 字符串数组
     * @return
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        if (strs == null || strs.length == 0){
            return new ArrayList<>();
        }
        Map<String, List<String>> map = new HashMap<>();
        int[] count = new int[26];
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.fill(count, 0);
            for (char c : chars) {
                count[c - 'a']++;
            }
//            Arrays.sort(chars);
//            String key = String.valueOf(chars);
            String key = Arrays.toString(count);
            if (!map.containsKey(key)){
                map.put(key, new ArrayList<>());
            }
            map.get(key).add(str);
        }
        return new ArrayList<>(map.values());
    }

    /**
     * 镜像二叉树 -- 递归
     * @param root - 根节点
     * @return
     */
    public boolean isSymmetric(TreeNode root) {
        if(root == null){
            return true;
        }
        return isMirror(root.left, root.right);
    }
    private boolean isMirror(TreeNode node1, TreeNode node2){
        if(node1 == null && node2 == null){
            return true;
        }else if(node1 == null || node2 == null){
            return false;
        }else if(node1.val != node2.val){
            return false;
        }
        return isMirror(node1.left, node2.right) && isMirror(node1.right, node2.left);
    }
    //迭代法
    public boolean isSymmetric2(TreeNode root){
        return isMirror2(root, root);
    }

    private boolean isMirror2(TreeNode node1, TreeNode node2) {
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(node1);
        q.offer(node2);
        while (!q.isEmpty()){
            node1 = q.poll();
            node2 = q.poll();
            if (node1 == null && node2 == null){
                continue;
            }else if (node1 == null || node2 == null || (node1.val != node2.val)){
                return false;
            }
            q.offer(node1.left);
            q.offer(node2.right);
            q.offer(node1.right);
            q.offer(node2.left);
        }
        return true;
    }

    /**
     * 方阵的幂次方
     * @param n - 幂次
     * @return
     */
    public int[][] ArrayOfN(int n){
        int[][] A = {{1,2,3},{1,3,2},{2,1,3}};
        int[][] tmp = {{1,0,0},{0,1,0},{0,0,1}};
        while (n > 0){
            tmp = ArrayProduct(A, tmp);
            n--;
        }
        return tmp;
    }
    /**
     * 矩阵乘法
     * @param A -- 矩阵A
     * @param B -- 矩阵B
     * @return
     */
    public int[][] ArrayProduct(int[][] A, int[][] B){
        int m = A.length;
        int n = A[0].length;
//        int p = B.length;      //注：n == p
        int q = B[0].length;
        int[][] tmp = new int[m][q];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < q; j++) {
                for (int k = 0; k < n; k++) {
                    tmp[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return tmp;
    }
    /**
     * 两大数相乘
     * @param s1
     * @param s2
     * @return
     */
    public String product(String s1, String s2){
        if (s1.equals("0") || s2.equals("0")){
            return "0";
        }
        String res = "0";
        int m = s1.length();
        int n = s2.length();
        for (int i = m - 1; i >= 0; i--) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < m - 1 - i; j++) {
                sb.append(0);
            }
            int num1 = s1.charAt(i) - '0';
            int carry = 0;
            for (int j = n - 1; j >= 0 || carry != 0; j--) {
                int num2 = j >= 0 ? s2.charAt(j) - '0' : 0;
                int product = (num1 * num2 + carry) % 10;
                sb.append(product);
                carry = (num1 * num2 + carry) / 10;
            }
            res = addString(sb.reverse().toString(), res);
        }
        return res;
    }
    /**
     * 两大数相加
     * @param s1
     * @param s2
     * @return
     */
    public String addString(String s1, String s2){
        StringBuilder sb = new StringBuilder();
        int m = s1.length();
        int n = s2.length();
        int carry = 0;
        for (int i = m - 1, j = n - 1; i >= 0 || j >= 0 || carry != 0; i--, j--) {
            int num1 = i >= 0 ? s1.charAt(i) - '0' : 0;
            int num2 = j >= 0 ? s2.charAt(j) - '0' : 0;
            int sum = (num1 + num2 + carry) % 10;
            sb.append(sum);
            carry = (num1 + num2 + carry) / 10;
        }
        return sb.reverse().toString();
    }
    /**
     * 旋转图像
     * @param matrix
     */
    public void rotate(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0){
            return;
        }
        int n = matrix.length;
        for (int i = 0; i <= (n - 1) / 2; i++) {
            for (int j = 0; j < n / 2; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - j][i];
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j];
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i];
                matrix[j][n - 1 - i] = tmp;
            }
        }
    }

    /**
     * 柱状图中最大的矩形 -- 单调栈
     * @param heights
     * @return
     */
    public int largestRectangleArea(int[] heights) {
        if (heights == null || heights.length == 0){
            return 0;
        }
        int maxArea = 0;
        int n = heights.length;
        Stack<Integer> s = new Stack<>();
        int[] left = new int[n];
        int[] right = new int[n];
        for (int i = 0; i < n; i++) {
            while (!s.isEmpty() && heights[s.peek()] >= heights[i]){
                s.pop();
            }
            left[i] = s.isEmpty() ? -1 : s.peek();
            s.push(i);
        }
        s.clear();
        for (int i = n - 1; i >= 0; i--) {
            while (!s.isEmpty() && heights[s.peek()] >= heights[i]){
                s.pop();
            }
            right[i] = s.isEmpty() ? n : s.peek();
            s.push(i);
        }
        for (int i = 0; i < n; i++) {
            maxArea = Math.max(maxArea, heights[i] * (right[i] - left[i] - 1));
        }
        return maxArea;
    }
    //暴力法
    public int largestRectangleArea2(int[] heights) {
        if (heights == null || heights.length == 0){
            return 0;
        }
        int maxArea = 0;
        int n = heights.length;
        for (int i = 0; i < n; i++) {
            int l = i;
            while (l > 0 && heights[l - 1] >= heights[i]){
                l--;
            }
            int r = i;
            while (r < n - 1 && heights[r + 1] >= heights[i]){
                r++;
            }
            maxArea = Math.max(maxArea, heights[i] * (r - l + 1));
        }
        return maxArea;
    }

    /**
     * 组合总和II -- 回溯法
     * @param candidates
     * @param target
     * @return
     */

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new LinkedList<>();
        if (candidates == null || candidates.length == 0){
            return res;
        }
        Arrays.sort(candidates);
        LinkedList<Integer> list = new LinkedList<>();
        recall(candidates, 0, target, list, res);
        return res;
    }

    private void recall(int[] candidates, int index, int target, LinkedList<Integer> list, List<List<Integer>> res) {
        if (target < 0) return;
        if (target == 0){
            list = new LinkedList<>(list);
            if (!res.contains(list)){
                res.add(list);
                return;
            }
        }
        int n = candidates.length;
        for (int i = index; i < n; i++) {
            if (target < candidates[i]) break;
            list.add(candidates[i]);
            recall(candidates, i + 1, target - candidates[i], list, res);
            list.removeLast();
        }
    }

    /**
     * 打家劫舍 III -- 有记忆式
     * @param root - 二叉树形式
     * @return
     */
    Map<TreeNode, Integer> map = new HashMap<>();
    public int robIII(TreeNode root) {
        if (root == null){
            return 0;
        }
        if (map.containsKey(root)){
            return map.get(root);
        }
        int notRoot = robIII(root.left) + robIII(root.right);
        int addRoot = root.val + (root.left == null ? 0 : robIII(root.left.left) + robIII(root.left.right)) +
                (root.right == null ? 0 : robIII(root.right.left) + robIII(root.right.right));
        int res = Math.max(notRoot, addRoot);
        map.put(root, res);
        return res;
    }

    public int robIII2(TreeNode root) {
        if (root == null){
            return 0;
        }
        int notRoot = robIII(root.left) + robIII(root.right);
        int addRoot = root.val + (root.left == null ? 0 : robIII(root.left.left) + robIII(root.left.right)) +
                (root.right == null ? 0 : robIII(root.right.left) + robIII(root.right.right));
        return Math.max(notRoot, addRoot);
    }

    /**
     * 打家劫舍 II
     * @param nums - 围成圆圈
     * @return
     */
    public int robII(int[] nums) {
        if(nums == null || nums.length == 0){
            return 0;
        }
        int n = nums.length;
        if(n == 1){
            return nums[0];
        }
        int pre1 = 0, cur1 = 0;
        for(int i = 0; i < n - 1; i++){
            int tmp = Math.max(pre1 + nums[i], cur1);
            pre1 = cur1;
            cur1 = tmp;
        }
        int pre2 = 0, cur2 = 0;
        for(int i = 1; i < n; i++){
            int tmp = Math.max(pre2 + nums[i], cur2);
            pre2 = cur2;
            cur2 = tmp;
        }
        return Math.max(cur1, cur2);
    }
    /**
     * 打家劫舍 -- 动态规划
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        if(nums == null || nums.length == 0){
            return 0;
        }
        int pre = 0, cur = 0;
        for (int num : nums) {
            int tmp = Math.max(pre + num, cur);
            pre = cur;
            cur = tmp;
        }
        return cur;
    }

    /**
     * 和可被K整除的子数组 - 同余定理
     * @param A
     * @param K
     * @return
     */
    public int subarraysDivByK(int[] A, int K) {
        if (A == null || A.length == 0){
            return 0;
        }
        int preSum = 0;
        int res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int num : A) {
            preSum += num;
            int modulus = (preSum % K + K) % K;
            int same = map.getOrDefault(modulus, 0);
            res += same;
            map.put(modulus, same + 1);
        }
        return res;
    }
    //超时
    public int subarraysDivByK2(int[] A, int K) {
        if (A == null || A.length == 0){
            return 0;
        }
        int preSum = 0;
        int res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < A.length; i++) {
            preSum += A[i];
            if (Math.abs(preSum) % K == 0){
                res++;
            }
            for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
                if (Math.abs(preSum - entry.getValue()) % K == 0 && (i - entry.getKey()) > 0){
                    res++;
                }
            }
            map.put(i, preSum);
        }
        return res;
    }

    /**
     * 寻找两个正序数组的中位数 -- 二分法
     * @param nums1
     * @param nums2
     * @return
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int len1 = nums1.length;
        int len2 = nums2.length;
        int len = len1 + len2;
        if (len % 2 == 1){
            return getKthElement(nums1, nums2, len / 2 + 1);
        }else {
            int midIndex1 = len / 2 - 1, minIndex2 = len / 2;
            return (getKthElement(nums1, nums2, midIndex1 + 1) + getKthElement(nums1, nums2, minIndex2 + 1)) / 2.0;
        }
    }

    private int getKthElement(int[] nums1, int[] nums2, int k) {
        int len1 = nums1.length, len2 = nums2.length;
        int index1 = 0, index2 = 0;
        while (true){
            if (index1 == len1){
                return nums2[index2 + k - 1];
            }
            if (index2 == len2){
                return nums1[index1 + k - 1];
            }
            if (k == 1){
                return Math.min(nums1[index1], nums2[index2]);
            }

            int mid = k / 2;
            int newIndex1 = Math.min(index1 + mid, len1) - 1;
            int newIndex2 = Math.min(index2 + mid, len2) - 1;
            if (nums1[newIndex1] <= nums2[newIndex2]){
                k -= (newIndex1 - index1 + 1);
                index1 = newIndex1 + 1;
            }else {
                k -= (newIndex2 - index2 + 1);
                index2 = newIndex2 + 1;
            }
        }
    }

    public double findMedianSortedArrays2(int[] nums1, int[] nums2) {
        int len1 = nums1.length;
        int len2 = nums2.length;
        int len = len1 + len2;
        int start1 = 0, start2 = 0;
        int midNum1 = -1, midNum2 = -1;
        for (int i = 0; i <= len / 2; i++) {
            midNum1 = midNum2;
            if (start1 < len1 && (start2 >= len2 || nums1[start1] < nums2[start2])){
                midNum2 = nums1[start1++];
            }else {
                midNum2 = nums2[start2++];
            }
        }
        if (len % 2 == 0){
            return (midNum1 + midNum2) / 2.0;
        }else {
            return midNum2;
        }
    }

    /**
     * 最小覆盖子串
     * @param s
     * @param t
     * @return
     */
    public String minWindow(String s, String t) {
        if (s == null || s.length() == 0 || t == null || t.length() == 0 || s.length() < t.length()){
            return "";
        }
        int[] sArr = new int[128];
        int[] tArr = new int[128];
        for (char c : t.toCharArray()) {
            tArr[c]++;
        }
        int i = 0, j = 0;
        String res = "";
        int count = 0;//当前字符数
        int minLen = s.length() + 1;
        while (j < s.length()){
            char c = s.charAt(j);
            sArr[c]++;
            if (tArr[c] > 0 && tArr[c] >= sArr[c]){
                count++;
            }
            while (count == t.length()){
                c = s.charAt(i);
                if (tArr[c] > 0 && tArr[c] >= sArr[c]){
                    count--;
                }
                if (j - i + 1 < minLen){
                    minLen = j - i + 1;
                    res = s.substring(i, j + 1);
                }
                sArr[c]--;
                i++;
            }
            j++;
        }
        return res;
    }
    /*public String minWindow(String s, String t) {
        if (s == null || s.length() == 0 || t == null || t.length() == 0 || s.length() < t.length()){
            return "";
        }
        int len = s.length();
        String res = s;
        List<Character> list = new ArrayList<>();
        for (char c : s.toCharArray()) {
            list.add(c);
        }
        List<Character> list1 = new ArrayList<>();
        for (char c : t.toCharArray()) {
            list1.add(c);
        }
        if (!list.containsAll(list1)){
            return "";
        }else {
            int i = 0, j = 0;
            while (j <= len){
                String s1 = s.substring(i, j);
                List<Character> list2 = new ArrayList<>();
                for (char c : s1.toCharArray()) {
                    list2.add(c);
                }
                if (list2.size() < list1.size() || !list2.containsAll(list1)){
                    j++;
                }else {
                    res = s1.length() < res.length() ? s1 : res;
                    i++;
                }
            }
        }
        return res;
    }*/

    /**
     * 从前序遍历与中序遍历序列构造二叉树
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) { ;
        if(preorder.length == 0 || inorder.length == 0){
            return null;
        }
        TreeNode root = new TreeNode(preorder[0]);
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        int inorderIndex = 0;
        for (int i = 1; i < preorder.length; i++) {
            int preVal = preorder[i];
            TreeNode node = stack.peek();
            if (node.val != inorder[inorderIndex]){
                node.left = new TreeNode(preVal);
                stack.push(node.left);
            }else {
                while (!stack.isEmpty() && stack.peek().val == inorder[inorderIndex]){
                    node = stack.pop();
                    inorderIndex++;
                }
                node.right = new TreeNode(preVal);
                stack.push(node.right);
            }
        }
        return root;
    }

    /**
     * 最长回文子串--中心扩散法
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        if(s == null || s.length() <= 1){
            return s;
        }
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int l1 = Palindrome(s, i, i);
            int l2 = Palindrome(s, i, i + 1);
            int maxLen = Math.max(l1, l2);
            if (maxLen > end - start){
                start = i - (maxLen - 1) / 2;
                end = i + maxLen / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    private int Palindrome(String s, int l, int r) {
        while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)){
            l--;
            r++;
        }
        return r - l - 1;
    }

    public class TreeNode{
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int val){
            this.val = val;
        }
    }
}
