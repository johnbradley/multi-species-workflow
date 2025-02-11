import argparse
from jinja2 import Environment
import pandas as pd
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse segmented directory and bioclip predictions path arguments.")
    parser.add_argument('--metadata', type=str, required=True, help='Path to the metadata csv')
    parser.add_argument('--segmented_dir', type=str, required=True, help='Path to the segmented directory')
    parser.add_argument('--predictions', type=str, required=True, help='Path to the bioclip predictions file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output html report file')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Segmented Directory: {args.segmented_dir}")
    print(f"Bioclip Predictions Path: {args.predictions}")
    print(f"Output Report Path: {args.output}")

    df = pd.read_csv(args.metadata)
    print(df)

    pred_df = pd.read_csv(args.predictions)
    print(pred_df)

    with open(args.segmented_dir + '/table_of_contents.json') as infile:
        table_of_contents = json.load(infile)
    print(table_of_contents)

    # Create a Jinja2 environment
    env = Environment()

    # Define an HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ title }}</title>
    </head>
    <body>
        <h1>{{ heading }}</h1>
        <ul>
            {% for item in items %}
                <li>{{ item }}</li>
            {% endfor %}
        </ul>
    </body>
    </html>
    """

    # Load the template
    template = env.from_string(html_template)

    # Define data to be rendered in the template
    data = {
        "title": "Sample Page",
        "heading": "Welcome to My Page",
        "items": ["Item 1", "Item 2", "Item 3"],
        "table_of_contents": table_of_contents,
    }

    # Render the template with data
    html_output = template.render(data)

    # Save the output to an HTML file
    with open(args.output, "w") as file:
        file.write(html_output)

    print(f"HTML file '{args.output}' generated successfully!")
