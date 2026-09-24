"""Generate the human/model-separated review summary from the current Excel table."""
from pathlib import Path
from utils.config import load_config
from utils.io import load_existing_table
from utils.reporting import summary_markdown


def create_recommended_summary(input_path, output_path, include_priorities=None):
    if include_priorities is not None:
        raise ValueError('read/skim priorities are retired; summaries retain all eligibility decisions.')
    rows=load_existing_table(input_path).to_dict('records')
    output=Path(output_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(summary_markdown(rows),encoding='utf-8')
    print(f'Saved review summary for {len(rows)} records to {output}')


def main():
    config=load_config('config.yaml')
    create_recommended_summary(config['sota_table_path'],config.get('review_summary_path','output/review_summary.md'))


if __name__=='__main__':
    main()
