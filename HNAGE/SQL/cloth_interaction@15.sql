SELECT *
FROM clothing_interaction15 
WHERE reviewerID IN (SELECT * FROM clothing_interaction15_usertrain) 
ORDER BY reviewerID,rank ASC 
;
WITH tmptable AS (
	SELECT DISTINCT(reviewerID)
	FROM clothing_interaction15
	WHERE reviewerID NOT IN (SELECT * FROM clothing_interaction15_usertrain) 
	-- LIMIT 2500
    LIMIT 200
	) 
SELECT *
FROM clothing_interaction15 
WHERE reviewerID IN (SELECT * FROM tmptable) 
ORDER BY reviewerID,rank ASC ;
;