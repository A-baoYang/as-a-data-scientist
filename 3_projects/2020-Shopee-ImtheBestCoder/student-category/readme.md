### 2020 I'm the Best Coder! (Student Category)

#### Performance

- Category: Logistics box size optimization
- Past Ranking: 
    - PublicBoard: 15 / 70 (21.4%)
    - PrivateBoard: 16 / 70 (22.9%)
- Past Score
    - PublicBoard: 0.53259 (Full Marks: 1.00000)
    - PrivateBoard: 0.53064 (Full Marks: 1.00000)
- Duration: 2020/11/21 15:00 - 2020/11/21 17:00
- [Kaggle Page](https://www.kaggle.com/c/iamthebestcoderstudent2020/leaderboard)

#### Topic

Optimize the packing operation by determining **the most viable carton box in the least space-wasteful way** for the warehouse since the larger one cost higher.

- Constraints:
    - Items **CANNOT** be placed oblique or be extruding from the box.

- Special Cases:
    - If 2 boxes are both the most viable ones to an order with the same volume, decide by the packing-box ID order. (priority: box id=1 > box id=2)
    - If there’s no one carton able to contain the items of an order, please specify the box_number as `UNFITTED`.
