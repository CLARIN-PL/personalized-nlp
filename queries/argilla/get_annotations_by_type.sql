SELECT *
FROM responses JOIN
    (SELECT records.id
    FROM
        (SELECT *
          FROM datasets
          WHERE name LIKE '%Correct_LLM_Answer%') ds
            JOIN
            records
            ON records.dataset_id = ds.id) rec_ds
    ON responses.record_id = rec_ds.id

-- %Task_Rank%
-- %Classify_Answer%
-- %Correct_LLM_Answer%
-- %Create_Answer%
