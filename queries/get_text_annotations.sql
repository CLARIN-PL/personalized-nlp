SELECT example_id, au.username, ee.text, question, answer, updated_at

FROM (SELECT example_id, user_id, text AS question, CAST(scale AS TEXT) AS answer, ls.updated_at AS updated_at
          FROM labels_scale ls
          JOIN label_types_scaletype lts ON ls.label_id = lts.id
          UNION
          SELECT example_id, user_id, question, text AS answer, updated_at
          FROM labels_textlabel) labels
      JOIN (SELECT id, username
          FROM auth_user) au ON au.id = labels.user_id
      JOIN (SELECT id, text
          FROM examples_example) ee ON labels.example_id = ee.id