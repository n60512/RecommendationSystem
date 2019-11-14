WITH tenReviewsUp AS ( 		
    SELECT reviewerID 		
    FROM review 		
    group by reviewerID 		
    HAVING COUNT(reviewerID) >= 20 		
    limit 2000 	) 

SELECT reviewerID 		
FROM review 		
WHERE reviewerID  NOT IN
(
    SELECT reviewerID 		
    FROM training_2000
)
group by reviewerID
HAVING COUNT(reviewerID) >= 20 		
limit 2000


SELECT RANK() OVER (
    PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC
    ) 
AS rank, review.`ID`, review.reviewerID , review.`asin`, 
        review.overall, review.reviewText, 
        review.unixReviewTime 
FROM review , metadata 
WHERE reviewerID 
IN (SELECT * FROM tenReviewsUp) 
AND review.`asin` = metadata.`asin` 
ORDER BY reviewerID,unixReviewTime ASC ;
