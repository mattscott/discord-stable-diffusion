import traceback
import requests
import asyncio
import discord
from discord.ext import commands
from typing import Optional
from io import BytesIO
from PIL import Image
from discord import option
from pytorch_lightning import seed_everything

from src.stablediffusion.text2image_diffusers import Text2Image

embed_color = discord.Colour.from_rgb(215, 195, 134)

class MyView(discord.ui.View): # Create a class called MyView that subclasses discord.ui.View
    def __init__(self, ctx, query: str, image: BytesIO, text2image_model: Text2Image, height: Optional[int]=512, width: Optional[int]=512, guidance_scale: Optional[float] = 7.0, steps: Optional[int] = 50, seed: Optional[int] = -1):
        super().__init__(timeout=None)
        self.ctx = ctx;
        self.query = query
        self.image = image
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.seed = seed
        self.text2image_model = text2image_model

    @discord.ui.button(custom_id="upscale", label="Upscale", row=0, style=discord.ButtonStyle.secondary, emoji="⏫")
    async def upscale_callback(self, button, interaction):
        await interaction.response.defer()
        try:
            embed = discord.Embed()
            embed.color = embed_color
            embed.set_footer(text=self.query)

            LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            res = self.image.resize((2048, 2048), resample=LANCZOS)

            with BytesIO() as buffer:
                res.save(buffer, 'PNG')
                buffer.seek(0)
                await self.ctx.send_followup(embed=embed, file=discord.File(fp=buffer, filename=f'{self.seed}-2048.png'))
        except Exception as e:
            embed = discord.Embed(title='Upscale failed', description=f'{e}\n{traceback.print_exc()}', color=embed_color)
            await self.ctx.send_followup(embed=embed)

        await interaction.response.send_message("Variation") # Send a message when the button is clicked

    @discord.ui.button(custom_id="variation", label="Make Variations", row=0, style=discord.ButtonStyle.secondary, emoji="🎯")
    async def variation_callback(self, button, interaction):
        await interaction.response.defer()
        try:
            embed = discord.Embed()
            embed.color = embed_color
            embed.set_footer(text=self.query)

            samples, seed = self.text2image_model.translation(self.query, self.image, 40, 0.0, 1, 1, 7.0, denoising_strength=0.4, seed=-1, height=self.height, width=self.width)

            with BytesIO() as buffer:
                samples[0].save(buffer, 'PNG')
                buffer.seek(0)
                myView = MyView(self.ctx, self.query, samples[0], self.text2image_model, self.height, self.width, self.guidance_scale, self.steps, seed)
                await self.ctx.send_followup(embed=embed, file=discord.File(fp=buffer, filename=f'{seed}.png'), view=myView)
        except Exception as e:
            embed = discord.Embed(title='Make Variations failed', description=f'{e}\n{traceback.print_exc()}', color=embed_color)
            await self.ctx.send_followup(embed=embed)

    @discord.ui.button(custom_id="doover", label="New Generation", row=0, style=discord.ButtonStyle.secondary, emoji="🔃")
    async def doover_callback(self, button, interaction):
        await interaction.response.defer()
        try:
            embed = discord.Embed()
            embed.color = embed_color
            embed.set_footer(text=self.query)

            rng_seed = seed_everything(self.seed)
            samples, seed = self.text2image_model.dream(self.query, self.steps, False, False, 0.0, 1, 1, self.guidance_scale, -1, self.height, self.width, False)

            with BytesIO() as buffer:
                samples[0].save(buffer, 'PNG')
                buffer.seek(0)
                myView = MyView(self.ctx, self.query, samples[0], self.text2image_model, self.height, self.width, self.guidance_scale, self.steps, seed)
                await self.ctx.send_followup(embed=embed, file=discord.File(fp=buffer, filename=f'{seed}.png'), view=myView)
        except Exception as e:
            embed = discord.Embed(title='New Generation failed', description=f'{e}\n{traceback.print_exc()}', color=embed_color)
            await self.ctx.send_followup(embed=embed)

    async def on_error(self, error, item, interaction):
        await interaction.response.send_message(str(error))

class StableCog(commands.Cog, name='Stable Diffusion', description='Create images from natural language.'):
    def __init__(self, bot):
        print(f'stablecog()')
        self.text2image_model = Text2Image()
        print(f'after Text2Image()')
        self.bot = bot

    @commands.slash_command(description='Create an image.')
    @option(
        'prompt',
        str,
        description = 'A prompt to condition the model with.',
        required=True,
    )
    @option(
        'height',
        int,
        description = 'Height of the generated image.',
        required = False,
        choices = [x for x in range(192, 832, 64)]
    )
    @option(
        'width',
        int,
        description = 'Width of the generated image.',
        required = False,
        choices = [x for x in range(192, 832, 64)]
    )
    @option(
        'guidance_scale',
        float,
        description = 'Classifier-Free Guidance scale',
        required = False,
    )
    @option(
        'steps',
        int,
        description = 'The amount of steps to sample the model',
        required = False,
        choices = [x for x in range(5, 105, 5)]
    )
    @option(
        'seed',
        int,
        description = 'The seed to use for reproduceability',
        required = False,
    )
    @option(
        'strength',
        float,
        description = 'The strength used to apply the prompt to the init_image/mask_image'
    )
    @option(
        'init_image',
        discord.Attachment,
        description = 'The image to initialize the latents with for denoising',
        required = False,
    )
    @option(
        'mask_image',
        discord.Attachment,
        description = 'The mask image to use for inpainting',
        required = False,
    )
    async def dream(self, ctx: discord.ApplicationContext, *, query: str, height: Optional[int]=512, width: Optional[int]=512, guidance_scale: Optional[float] = 7.0, steps: Optional[int] = 50, seed: Optional[int] = -1, strength: Optional[float]=0.8, init_image: Optional[discord.Attachment] = None, mask_image: Optional[discord.Attachment] = None):
        print(f'Request -- {ctx.author.name}#{ctx.author.discriminator} -- Prompt: {query}')
        await ctx.defer()
        embed = discord.Embed()
        embed.color = embed_color
        embed.set_footer(text=query)
        try:
            if (init_image is None) and (mask_image is None):
                samples, seed = self.text2image_model.dream(query, steps, False, False, 0.0, 1, 1, guidance_scale, seed, height, width, False)
            elif (init_image is not None):
                image = Image.open(requests.get(init_image.url, stream=True).raw).convert('RGB')
                samples, seed = self.text2image_model.translation(query, image, steps, 0.0, 0, 0, guidance_scale, strength, seed, height, width)
            else:
                image = Image.open(requests.get(init_image.url, stream=True).raw).convert('RGB')
                mask = Image.open(requests.get(mask_image.url, stream=True).raw).convert('RGB')
                samples, seed = self.text2image_model.inpaint(query, image, mask, steps, 0.0, 1, 1, guidance_scale, denoising_strength=strength, seed=seed, height=height, width=width)

            with BytesIO() as buffer:
                samples[0].save(buffer, 'PNG')
                buffer.seek(0)
                # await ctx.followup.send(embed=embed, file=discord.File(fp=buffer, filename=f'{seed}.png'))
                myView = MyView(ctx, query, samples[0], self.text2image_model, height, width, guidance_scale, steps, seed)
                await ctx.send_followup(embed=embed, file=discord.File(fp=buffer, filename=f'{seed}.png'), view=myView)

        except Exception as e:
            embed = discord.Embed(title='txt2img failed', description=f'{e}\n{traceback.print_exc()}', color=embed_color)
            await ctx.followup.send(embed=embed)
    
    @commands.slash_command(description='Test what an image looks like from the model\'s perspective')
    async def vae(self, ctx: discord.ApplicationContext, *, image_url: str, height: Optional[int]=512, width: Optional[int]=512):
        await ctx.defer()
        try:
            image = Image.open(requests.get(image_url, stream=True).raw).convert('RGBA')
            samples = self.text2image_model.vae_test(image, height, width)
            with BytesIO() as buffer:
                samples[0].save(buffer, 'PNG')
                buffer.seek(0)
                await ctx.followup.send(file=discord.File(fp=buffer, filename=f'decoded.png'))
        except Exception as e:
            embed = discord.Embed(title='vae failed', description=f'{e}\n{traceback.print_exc()}', color=embed_color)
            await ctx.followup.send(embed=embed)

def setup(bot):
    bot.add_cog(StableCog(bot))
