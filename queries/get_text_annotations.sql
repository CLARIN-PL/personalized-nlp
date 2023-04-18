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

WHERE au.username IN ('1120', '1160', '1210', '1211', '1215', '1257', '1314', '1436', '1638',
                      '1847', '2238', '2290', '2621', '3014', '3163', '3465', '3758', '4060',
                      '4073', '4384', '4411', '4444', '4482', '4520', '4643', '4760', '4963',
                      '5282', '5362', '5418', '5529', '5955', '6353', '6513', '6556', '6734',
                      '6751', '6818', '6954', '6992', '7102', '7597', '7621', '7667', '7691',
                      '7727', '8194', '8241', '8342', '8533', '8583', '8858', '8889', '9178',
                      '9224', '9327', '9458', '9672', '2525', '3884', '9707', '4339', '1640')

-- WHERE question LIKE '%Ze względu na co obraża%'
-- WHERE question LIKE '%W jaki sposób%'
-- WHERE question LIKE '%Autor śmieje się z%'
-- WHERE question LIKE '%Charakter humoru%'
