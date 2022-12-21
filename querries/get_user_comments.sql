SELECT ee.text, ec.updated_at, au.username, ec.text

FROM auth_user au JOIN examples_comment ec on au.id = ec.user_id
    JOIN examples_example ee on ec.example_id = ee.id