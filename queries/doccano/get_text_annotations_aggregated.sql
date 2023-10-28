SELECT ls_lts.example_id, au.username, ee.text, label_text_name, label_text_value, label_scale_name, label_scale_value

-- added DISTINCT to filter out answers changed over time

FROM (SELECT example_id, user_id, array_agg(ls.text) AS label_scale_name, array_agg(ls.scale) AS label_scale_value
          FROM ((SELECT DISTINCT example_id, user_id, label_id, scale
                FROM labels_scale) ls_distinct
            JOIN (SELECT id, text
                  FROM label_types_scaletype) lts ON ls_distinct.label_id = lts.id) ls
          GROUP BY example_id, user_id) ls_lts
    JOIN (SELECT example_id, user_id, array_agg(question) AS label_text_name, array_agg(text) AS label_text_value
          FROM (SELECT DISTINCT example_id, user_id, question, text
                FROM labels_textlabel) lt_distinct
          GROUP BY example_id, user_id) lt ON ls_lts.example_id = lt.example_id AND ls_lts.user_id = lt.user_id
    JOIN (SELECT id, username
          FROM auth_user) au ON au.id = ls_lts.user_id
    JOIN (SELECT id, text
      FROM examples_example) ee ON ls_lts.example_id = ee.id